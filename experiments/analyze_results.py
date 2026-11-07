"""Recompute report statistics from completed evaluations using only stdlib."""
import argparse
import json
from pathlib import Path
import statistics


def describe(values):
    return {'mean':statistics.mean(values),
            'sample_sd':statistics.stdev(values) if len(values)>1 else None,
            'sample_variance':statistics.variance(values) if len(values)>1 else None}


def analyze(study):
    state=json.loads((study/'study_status.json').read_text())
    if state['status']!='complete' or not state.get('evaluation_finalized'):
        raise RuntimeError('Generation and frozen scoring must finish first')
    planned={(s,b) for s in state['seeds'] for b in state['iteration_budgets']}
    assert {(r['seed'],r['n_iters']) for r in state['completed_runs']}==planned
    n=len(state['question_ids']);runs=[];points=[]
    for meta in state['completed_runs']:
        ev=json.loads((study/f"seed{meta['seed']}_iters{meta['n_iters']}"/'offline_evaluation.json').read_text())
        assert ev['exit_code']==0 and len(ev['examples'])==n
        assert [e['id'] for e in ev['examples']]==state['question_ids']
        a=sum(e['correct'] for e in ev['examples']);b=sum(e['extended_correct'] for e in ev['examples'])
        runs.append({'seed':meta['seed'],'n_iters':meta['n_iters'],
                     'baseline_correct':a,'modified_correct':b,
                     'baseline_accuracy':a/n,'modified_accuracy':b/n,'paired_difference':(b-a)/n})
        points.extend(ev['examples'])
    by_budget={}
    for budget in state['iteration_budgets']:
        group=[r for r in runs if r['n_iters']==budget]
        by_budget[str(budget)]={
            'baseline':describe([r['baseline_accuracy'] for r in group]),
            'modified':describe([r['modified_accuracy'] for r in group]),
            'paired_difference':describe([r['paired_difference'] for r in group])}
    rest=[p for p in points if p['index']!=0]
    result={'unique_questions':n,'task_executions':len(points),
            'manual_review_status':state.get('manual_review_status','pending'),
            'runs':runs,'by_iteration_budget':by_budget,
            'changed_predictions':sum(p['prediction']!=p['extended_prediction'] for p in points),
            'incorrect_to_correct':sum(not p['correct'] and p['extended_correct'] for p in points),
            'correct_to_incorrect':sum(p['correct'] and not p['extended_correct'] for p in points),
            'excluding_development_question_0':{
                'unique_questions':max(n-1,0),'task_executions':len(rest),
                'baseline_correct':sum(p['correct'] for p in rest),
                'modified_correct':sum(p['extended_correct'] for p in rest)}}
    (study/'report_statistics.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path)
    print(json.dumps(analyze(p.parse_args().study),indent=2))
