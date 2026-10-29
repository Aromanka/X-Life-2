"""Require explicit task anchors and measured labels before CBG/night experiments."""
from datetime import datetime
import math
import torch
from dataset import event_timestamp


def measured_grid(points,origin,minutes,future):
    if minutes<=0 or minutes%5: raise ValueError('Task windows must use 5-minute steps')
    values=torch.zeros(minutes//5);mask=torch.zeros(minutes//5,dtype=torch.bool)
    for timestamp,value in points:
        time=event_timestamp('',timestamp)
        if time is None or not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError('Invalid measured task point')
        offset=(time-origin).total_seconds()/60
        if (0<offset<=minutes) if future else (-minutes<=offset<0):
            index=math.ceil(offset/5)-1 if future else math.floor(offset/5)+len(values)
            if mask[index]: raise ValueError('Multiple labels in a task slot require an explicit aggregation policy')
            values[index]=value;mask[index]=True
    if not mask.any(): raise ValueError('No measured values in task window')
    return values,mask


def task_window(record):
    task=record['task']
    if task not in ('cbg','night'): raise ValueError('Expected cbg or night')
    origin=datetime.fromisoformat(record['origin'])
    if record.get('glucose_unit') not in ('mg/dL','mmol/L'): raise ValueError('Task glucose units required')
    if not record.get('label_source'): raise ValueError('Measured target source required')
    if task=='night':
        if record.get('bedtime')!=record['origin'] or not record.get('wake_time'): raise ValueError('Night requires recorded bedtime and wake time')
        span=(datetime.fromisoformat(record['wake_time'])-origin).total_seconds()/60
        if span!=record['target_minutes']: raise ValueError('Night horizon must match recorded wake time')
    history,hm=measured_grid(record['history_points'],origin,record['history_minutes'],False)
    target,tm=measured_grid(record['target_points'],origin,record['target_minutes'],True)
    return dict(task=task,subject_id=record['subject_id'],origin=record['origin'],history=history,history_mask=hm,
                target=target,target_mask=tm,glucose_unit=record['glucose_unit'],label_source=record['label_source'])
