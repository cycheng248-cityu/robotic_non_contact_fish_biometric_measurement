"""Offline yaw ACK analysis for Spyder; no cameras, network, or motors used.
One explicit pulse request = one candidate event. An issued event has >=1 nonzero TX.
Success = >=1 accepted MATCHED_ACK with sequence, echo and peer matching within timeout.
Assumes one complete, monotonically timed session per CSV. Ambiguity raises an error.
"""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

REQUEST_KINDS = {'PULSE_REQUEST', 'PULSEREQUEST'}

def parse_command(payload, ack=False):
    text = str(payload).strip()
    if ack:
        if not text.startswith('ACK:'): raise ValueError('ACK prefix missing')
        text = text[4:]
    parts = text.split(',')
    if len(parts)!=4: raise ValueError('Expected sequence,forward,yaw,vertical')
    seq = str(int(parts[0].strip()))
    values = np.array([float(v) for v in parts[1:]])
    if not np.isfinite(values).all(): raise ValueError('Nonfinite command')
    return seq, values

def parse_peer(value):
    peer = ast.literal_eval(str(value))
    if not isinstance(peer,tuple) or len(peer)!=2: raise ValueError('Invalid peer')
    return str(peer[0]),int(peer[1])

def analyze_log(csv_path, fish_id, run_id=None, ack_window_s=0.35,
                echo_tolerance=0.0006):
    """Return run summary, event audit, packet audit, and rejected-ACK audit.
    ACK_window is per transmitted packet, not per request or pulse duration.
    Missing ACK at an incomplete observation boundary is classified unknown.
    MATCHED_ACK is required: raw/rejected ACKs never become accepted successes.
    """
    if not np.isfinite(ack_window_s) or ack_window_s<=0: raise ValueError('Invalid ACK window')
    path=Path(csv_path).expanduser().resolve()
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    required={'monotonic','kind','sequence','yaw','reason','peer','payload'}
    if not required.issubset(df.columns): raise ValueError('Missing columns: '+str(required-set(df.columns)))
    t=pd.to_numeric(df.monotonic,errors='raise').to_numpy(float)
    if not len(t) or not np.isfinite(t).all() or (np.diff(t)<0).any():
        raise ValueError('Empty log or invalid/nonmonotonic timestamps; split sessions first')
    df['time']=t
    run_id=str(run_id) if run_id is not None else path.parent.name
    base={'fish_id':str(fish_id),'run_id':run_id,'source_csv':str(path)}
    events=[];packets=[];acks=[];current=None; seen=set()
    for row_idx,r in df.iterrows():
        kind=r['kind'].strip().upper()
        if kind in REQUEST_KINDS:
            yaw=float(r['yaw'])
            if not np.isfinite(yaw) or abs(yaw)<=1e-9: raise ValueError('Invalid pulse request yaw')
            if current is not None and events[current]['end_s'] is None:
                raise ValueError('Overlapping requests without a neutral TX; inspect log')
            current=len(events)
            events.append({**base,'event_id':current+1,'request_s':r.time,'requested_yaw':yaw,
                           'end_s':None,'ACK_stale_fault':False,'packet_indices':[]})
        elif kind=='FAULT' and current is not None:
            if events[current]['end_s'] is None and ('ACK_stale' in r.reason or 'ACK_stale' in r.payload):
                events[current]['ACK_stale_fault']=True
        elif kind=='TX':
            seq,values=parse_command(r.payload)
            if str(int(r.sequence))!=seq: raise ValueError('TX sequence column/payload mismatch')
            if seq in seen: raise ValueError('Repeated TX sequence: cannot unambiguously compute latency')
            seen.add(seq)
            if abs(values[1])>1e-9:
                if current is None or events[current]['end_s'] is not None:
                    raise ValueError('Nonzero TX without an active explicit pulse request')
                if abs(values[1]-events[current]['requested_yaw'])>echo_tolerance:
                    raise ValueError('Yaw direction/magnitude changes within event')
                events[current]['packet_indices'].append(len(packets))
                packets.append({**base,'event_id':current+1,'sequence':seq,'tx_s':r.time,
                    'yaw':float(values[1]),'_values':values,'_peer':parse_peer(r.peer),
                    'ack_s':None,'latency_ms':np.nan})
            elif current is not None and events[current]['end_s'] is None:
                events[current]['end_s']=r.time
        elif kind=='MATCHED_ACK': acks.append((row_idx,r))
    lookup={p['sequence']:p for p in packets}
    invalid=[]
    for row_idx,r in acks:
        try:
            seq,values=parse_command(r.payload,ack=True)
            if seq not in lookup: continue  # Neutral ACKs cannot qualify.
            p=lookup[seq]
            if str(int(r.sequence))!=seq: raise ValueError('ACK sequence column mismatch')
            dt=float(r.time-p['tx_s'])
            if parse_peer(r.peer)!=p['_peer']: raise ValueError('wrong peer')
            if np.any(abs(values-p['_values'])>echo_tolerance): raise ValueError('echo mismatch')
            if dt<0 or dt>ack_window_s: raise ValueError('outside ACK window')
            if p['ack_s'] is not None: raise ValueError('duplicate ACK')
            p['ack_s']=float(r.time);p['latency_ms']=dt*1000
        except (ValueError,SyntaxError,TypeError) as exc:
            invalid.append({**base,'row_index':row_idx,'payload':r.payload,'issue':str(exc)})
    for p in packets:
        p['status']='acknowledged' if p['ack_s'] is not None else (
            'not_acknowledged' if t[-1]>=p['tx_s']+ack_window_s else 'unknown_log_ended')
        del p['_values'],p['_peer']
    out_events=[]
    for e in events:
        indices=e.pop('packet_indices'); ps=[packets[i] for i in indices]
        good=[p for p in ps if p['status']=='acknowledged']
        pending=[p for p in ps if p['status']=='unknown_log_ended']
        status=('not_issued' if not ps else 'acknowledged' if good else
                'unknown_log_ended' if pending else 'not_acknowledged')
        e.update(status=status,yaw_packets=len(ps),acknowledged_packets=len(good),
                 unknown_packets=len(pending),
                 first_tx_s=ps[0]['tx_s'] if ps else np.nan,
                 first_ack_s=min(p['ack_s'] for p in good) if good else np.nan)
        e['request_to_first_ACK_ms']=(e['first_ack_s']-e['request_s'])*1000
        e['first_ACK_packet_latency_ms']=(min(good,key=lambda p:p['ack_s'])['latency_ms'] if good else np.nan)
        out_events.append(e)
    n=sum(e['status']!='not_issued' for e in out_events)
    yes=sum(e['status']=='acknowledged' for e in out_events)
    no=sum(e['status']=='not_acknowledged' for e in out_events)
    unk=n-yes-no
    summary={**base,'ack_window_s':ack_window_s,'pulse_requests':len(events),
        'not_issued_requests':len(events)-n,'issued_events':n,'acknowledged_events':yes,
        'unacknowledged_events':no,'unknown_events':unk,
        'event_ACK_rate_percent':100*yes/n if n and not unk else np.nan,
        'observed_event_ACK_lower_bound_percent':100*yes/n if n else np.nan,
        'event_ACK_upper_bound_percent':100*(yes+unk)/n if n else np.nan,
        'events_with_ACK_stale_fault':sum(e['ACK_stale_fault'] for e in out_events if e['status']!='not_issued'),
        'yaw_TX_packets':len(packets),'acknowledged_yaw_packets':sum(p['status']=='acknowledged' for p in packets),
        'unknown_yaw_packets':sum(p['status']=='unknown_log_ended' for p in packets),
        'pulse_reject_rows':int(df.kind.str.upper().isin(['PULSE_REJECT','PULSEREJECT']).sum()),
        'invalid_matched_yaw_ACK_rows':len(invalid)}
    return {'run_summary':summary,'events':pd.DataFrame(out_events),
            'packets':pd.DataFrame(packets),'ack_issues':pd.DataFrame(invalid)}

def aggregate_summaries(runs, group_by=None):
    count_cols=['pulse_requests','not_issued_requests','issued_events','acknowledged_events',
        'unacknowledged_events','unknown_events','events_with_ACK_stale_fault',
        'yaw_TX_packets','acknowledged_yaw_packets','unknown_yaw_packets','pulse_reject_rows',
        'invalid_matched_yaw_ACK_rows']
    if group_by: out=runs.groupby(group_by,dropna=False)[count_cols].sum().reset_index()
    else: out=pd.DataFrame([{c:int(runs[c].sum()) for c in count_cols}])
    n=out.issued_events.replace(0,np.nan)
    out['event_ACK_rate_percent']=(100*out.acknowledged_events/n).where(out.unknown_events==0)
    out['observed_event_ACK_lower_bound_percent']=100*out.acknowledged_events/n
    out['event_ACK_upper_bound_percent']=100*(out.acknowledged_events+out.unknown_events)/n
    out['yaw_packet_ACK_rate_percent']=(100*out.acknowledged_yaw_packets/out.yaw_TX_packets.replace(0,np.nan)).where(out.unknown_yaw_packets==0)
    return out

def analyze_fish_logs(log_specs, output_dir=None, ack_window_s=0.35):
    """log_specs: list of dicts with fish_id, csv_path, optional run_id.
    Multiple runs per fish permitted. Exact duplicate paths are rejected.
    Uses pooled counts, NOT an average of percentages. Invalid files raise errors.
    """
    if not log_specs: raise ValueError('Provide at least one log')
    paths=[str(Path(s['csv_path']).expanduser().resolve()) for s in log_specs]
    if len(paths)!=len(set(paths)): raise ValueError('Same log supplied twice')
    analyses=[analyze_log(s['csv_path'],s['fish_id'],s.get('run_id'),ack_window_s) for s in log_specs]
    runs=pd.DataFrame([a['run_summary'] for a in analyses])
    if runs[['fish_id','run_id']].duplicated().any(): raise ValueError('Use distinct run_id values within each fish')
    result={'run_summary':runs,'fish_summary':aggregate_summaries(runs,'fish_id'),
            'pooled_summary':aggregate_summaries(runs)}
    for key in ['events','packets','ack_issues']:
        result[key]=pd.concat([a[key] for a in analyses],ignore_index=True)
    if output_dir is not None:
        out=Path(output_dir).expanduser();out.mkdir(parents=True,exist_ok=True)
        for key,table in result.items(): table.to_csv(out/(key+'.csv'),index=False)
        (out/'analysis_settings.json').write_text(json.dumps({'ack_window_s':ack_window_s,
            'definition':'At least one nonzero TX receives a qualifying MATCHED_ACK within the per-packet window.',
            'physical_yaw_verified':False,'logs':paths},indent=2))
    return result

# SPYDER USER SETTINGS: replace paths and add one entry for each included run.
if __name__=='__main__':
    LOGS=[
        {'fish_id':'fish_1','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_1\udp_link.csv"},
        {'fish_id':'fish_2','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_2\udp_link.csv"},
        {'fish_id':'fish_3','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_3\udp_link.csv"},
        {'fish_id':'fish_4','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_4\udp_link.csv"},
        {'fish_id':'fish_5','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_5\udp_link.csv"},
        {'fish_id':'fish_6','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_6\udp_link.csv"},
        {'fish_id':'fish_7','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_7\udp_link.csv"},
        {'fish_id':'fish_8','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_8\udp_link.csv"},
        {'fish_id':'fish_9','run_id':'run_1',
         'csv_path':r"D:\2nd_project_AUV_deepwater_fish_size_estimation\AUV_length_data\fish_9\udp_link.csv"},
     
    ]
    ACK_WINDOW_S=0.35  # Verify against the controller version used for these runs.
    OUTPUT_DIR=Path(__file__).resolve().parent/'yaw_ack_results'
    results=analyze_fish_logs(LOGS,output_dir=OUTPUT_DIR,ack_window_s=ACK_WINDOW_S)
    fish_summary=results['fish_summary']
    run_summary=results['run_summary']
    pooled_summary=results['pooled_summary']
    event_audit=results['events']
    packet_audit=results['packets']
    ack_issues=results['ack_issues']
    print(fish_summary.to_string(index=False))
    print(pooled_summary.to_string(index=False))
