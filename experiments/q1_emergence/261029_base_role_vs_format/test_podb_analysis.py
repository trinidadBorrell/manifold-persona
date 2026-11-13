"""Offline checks for the frozen Pod B estimands; no model or real arrays."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('podb_analysis_tests_target', HERE / 'analysis.py')
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def planted(kind, R=6, F=4, Q=8, W=5, D=7):
    rng = np.random.default_rng(41)
    shape = (R, F, Q, W, D)
    if kind == 'role':
        x = rng.normal(size=(R, 1, 1, 1, D))
    elif kind == 'format':
        x = rng.normal(size=(1, F, 1, 1, D))
    elif kind == 'role_format':
        x = rng.normal(size=(R, F, 1, 1, D))
        x = x - x.mean(0, keepdims=True) - x.mean(1, keepdims=True) + x.mean((0, 1), keepdims=True)
    return np.broadcast_to(x, shape).copy()


@pytest.mark.parametrize('kind', ['role', 'format', 'role_format'])
def test_balanced_ss_recovers_planted_terms(kind):
    out = A.variance_split(planted(kind))
    assert out['shares'][kind] == pytest.approx(1)
    assert sum(out['shares'].values()) == pytest.approx(1)
    for key, value in out['shares'].items():
        if key != kind:
            assert abs(value) < 1e-20


def test_balanced_ss_orthogonal_partition_on_arbitrary_grid():
    x = np.random.default_rng(1).normal(size=(6, 4, 8, 5, 7))
    out = A.variance_split(x)
    assert len(out['ss']) == 8
    assert sum(out['ss'].values()) == pytest.approx(out['total_ss'], rel=1e-12)
    assert out['max_abs_cross_term'] < 1e-9


def row_grid(x, nrep=3):
    rows, xx = [], []
    for r, f, q, w in np.ndindex(x.shape[:4]):
        for rep in range(nrep):
            rows.append(dict(role=f'r{r}', format=f'f{f}', question=q, wording=w, replicate=rep,
                             valid=True, reply='the reply', kept_ids=[1], stop_reason='limit'))
            xx.append(x[r, f, q, w])
    return rows, np.array(xx)


def test_prompt_means_equal_weight_despite_unequal_replicates():
    x = planted('role', R=4, Q=4)
    rows, xx = row_grid(x)
    # One valid replicate in one prompt; its mean remains that prompt's original value.
    rows[0]['valid'] = rows[1]['valid'] = False
    xx[:2] = np.nan
    c = A.prompt_cells(rows, xx, [f'r{i}' for i in range(4)], [f'f{i}' for i in range(4)], list(range(4)))
    assert c['counts'][0, 0, 0, 0] == 1
    np.testing.assert_allclose(c['means'], x)
    assert A.variance_split(c['means'])['shares']['role'] == pytest.approx(1)


def test_missing_prompt_drops_question_everywhere_and_enforces_support():
    x = planted('role', R=4, Q=4)
    x[0, 0, 0, 0] = np.nan
    assert A.complete_grid(x, min_questions=4)['status'] == 'not_estimable'
    out = A.complete_grid(x, min_questions=3)
    assert out['kept_questions'] == [1, 2, 3]
    assert out['grid'].shape == (4, 4, 3, 5, 7)
    assert A.complete_grid(np.broadcast_to(np.ones(1), (10,4,29,5,1)).copy())['status'] == 'not_estimable'


def test_splits_hold_out_format_question_and_wording_together():
    rows, xx = row_grid(planted('role', R=4))
    c = A.prompt_cells(rows, xx, [f'r{i}' for i in range(4)], [f'f{i}' for i in range(4)], list(range(8)))
    p = A.flatten_cells(c)
    splits = A.make_splits(p, 4, n=12, seed=1)
    for f, folds in splits.items():
        for tr, te in folds:
            assert set(p['f'][te]) == {f}
            assert f not in p['f'][tr]
            assert not set(p['q'][tr]) & set(p['q'][te])
            assert not set(p['w'][tr]) & set(p['w'][te])
            assert len(set(p['y'][tr])) == len(set(p['y'][te])) == 4


def test_block_permutation_keeps_all_wordings_formats_together_per_question():
    rows, xx = row_grid(planted('role', R=4))
    p = A.flatten_cells(A.prompt_cells(rows, xx, [f'r{i}' for i in range(4)], [f'f{i}' for i in range(4)], list(range(8))))
    yp = A.block_permute(p['y'], p['q'], np.random.default_rng(7))
    for q in range(8):
        mapping = [np.unique(yp[(p['q'] == q) & (p['y'] == r)]).tolist() for r in range(4)]
        assert sorted(v[0] for v in mapping) == list(range(4))
        assert all(len(v) == 1 for v in mapping)


def test_unseen_format_decoder_is_at_chance_for_format_only_signal():
    rows, xx = row_grid(planted('format', R=4))
    p = A.flatten_cells(A.prompt_cells(rows, xx, [f'r{i}' for i in range(4)], [f'f{i}' for i in range(4)], list(range(8))))
    for folds in A.make_splits(p, 4, n=5).values():
        out = A.decode(p['X'], p, folds)
        assert out['accuracy'] == pytest.approx(.25)


def test_full_refit_null_has_plus_one_p_and_recovers_role_signal(tmp_path):
    rows, xx = row_grid(planted('role', R=4))
    p = A.flatten_cells(A.prompt_cells(rows, xx, [f'r{i}' for i in range(4)], [f'f{i}' for i in range(4)], list(range(8))))
    folds = A.make_splits(p, 4, n=5)[0]
    out = A.decode(p['X'], p, folds)
    assert out['accuracy'] == pytest.approx(1)
    result = A.decode_null(p['X'], p, folds, n=9, seed=4, checkpoint=tmp_path / 'null.json')
    assert result['p'] == pytest.approx((1+sum(v>=result['observed'] for v in result['null']))/10)
    assert len(json.loads((tmp_path/'null.json').read_text())['values']) == 9
    assert result == A.decode_null(p['X'], p, folds, n=9, seed=4, checkpoint=tmp_path / 'null.json')


def test_cross_format_maps_agree_for_common_role_geometry():
    x = planted('role')
    halves = A.make_halves(8, 5, n=7)
    d = A.map_distances(x, halves)
    stat = A.map_statistic(d)
    assert stat['mean'] == pytest.approx(1)
    assert len(stat['pairs']) == 6
    assert all(v == pytest.approx(1) for v in stat['reliability'].values())


def test_joint_null_uses_one_permutation_per_format_in_all_six_pairs():
    x = np.random.default_rng(2).normal(size=(6,4,8,5,7))
    d = A.map_distances(x, A.make_halves(8,5,n=4))
    draws = A.joint_map_null(d, n=5, seed=8)
    rng = np.random.default_rng(8)
    for value in draws:
        perms = np.array([np.arange(6)] + [rng.permutation(6) for _ in range(3)])
        dp = np.stack([d[f][..., perms[f], :][..., perms[f]] for f in range(4)])
        assert value == pytest.approx(A.map_statistic(dp)['mean'])


def test_holm_adjustment_handles_missing_decision_conservatively():
    assert A.holm({'D1':.004,'D2':.02,'D3':.03}) == pytest.approx({'D1':.012,'D2':.04,'D3':.04})
    assert A.holm({'D1':None,'D2':.004,'D3':.02}) == {'D1':None,'D2':.012,'D3':.04}


def test_nonfinite_bootstrap_never_silently_discards_draws():
    out = A.interval(1.0, [1.0, np.nan, 1.0])
    assert out['lo'] is out['hi'] is None
    assert out['n_undefined'] == 1


def test_g6_synthetic_runs_all_required_checks():
    out = A.synthetic_checks()
    assert out['pass']
    assert {'variance','missing_unequal_replicates','format_only_decoding','joint_null_calibration'} <= out.keys()


def test_smoke_parameters_are_never_confirmatory():
    assert A.confirmatory_parameters(n_boot=2000,n_perm=9999,n_splits=50,n_halves=50)
    assert not A.confirmatory_parameters(n_boot=20,n_perm=9999,n_splits=50,n_halves=50)


def test_ngram_vocabulary_fits_training_replies_only_and_prompt_means():
    texts=['alpha alpha common','alpha common','testonly beta','testonly gamma']
    groups=[[0,1],[2,3]]
    feat=A.text_featurizer(texts,groups,'word')
    tr,te=feat(np.array([0]),np.array([1]))
    assert sparse_shape(tr)==sparse_shape(te)
    assert np.linalg.norm(te.toarray()) == pytest.approx(0)


def sparse_shape(x):
    return x.shape


def test_all_valid_fit_uses_unequal_replicate_weights_without_inventing_term_shares():
    x=planted('role',R=4,Q=4)
    rows,xx=row_grid(x)
    rows[0]['valid']=rows[1]['valid']=False;xx[:2]=np.nan
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(4)))
    out=A.all_valid_fit(c)
    assert out['residual_ss'] < 1e-20
    assert out['n_rows']==len(rows)-2
    assert 'shares' not in out


def test_context_mapping_rejects_duplicate_or_absent_prompt_keys():
    keys=[['r0','f0',0,0],['r0','f0',1,0]]
    x=np.ones((2,3))
    assert A.context_cells(x,keys,['r0'],['f0'],[0])['counts'].sum()==2
    with pytest.raises(A.AnalysisError,match='duplicate'):
        A.context_cells(x,[keys[0],keys[0]],['r0'],['f0'],[0])


def test_gate_evidence_cannot_pass_with_wrong_counts_or_missing_reconstruction():
    gate={'passed':True,'gates':{f'G{i}':{'pass':True} for i in range(1,6)}}
    with pytest.raises(A.AnalysisError):
        A.validate_gate_evidence(gate,n_rows=24000,n_valid=23990)


def test_synthetic_only_cli_does_not_need_run_or_downloads(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,str(HERE/'analysis.py'),'--synthetic-only','--out',str(tmp_path/'g6')],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'g6'/'G6.json').read_text())
    assert report['pass']


def test_crossed_map_bootstrap_matches_direct_vectors_with_duplicate_roles():
    x=np.random.default_rng(9).normal(size=(6,4,8,5,7))
    halves=A.make_halves(8,5,n=4,seed=2)
    rr=np.array([0,0,1,3,3,5]);qq=np.array([0,0,1,2,3,5,6,7])
    result=A.d3_bootstrap(x,halves,[(rr,qq)])
    d=A.map_distances(x,halves,np.bincount(qq,minlength=8),rr)
    _,pos=np.unique(rr,return_inverse=True)
    i,j=np.triu_indices(6,1);keep=rr[i]!=rr[j]
    expected=A.map_statistic(d,(pos[i[keep]],pos[j[keep]]))
    assert result['values'][0]==pytest.approx(expected['mean'],abs=1e-12)


def test_report_only_decoder_marks_missing_role_fold_unavailable():
    x=planted('role',R=4)
    x[0,0]=np.nan
    rows,xx=row_grid(x)
    for i in range(len(rows)):
        rows[i]['valid']=bool(np.isfinite(xx[i]).all())
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8)))
    result=A.descriptive_decoding(c,n_splits=4,n_boot=10)
    assert result['status']=='not_estimable'
    assert 'role' in result['reason'].lower()


def test_raw_row_residual_includes_within_prompt_replicate_variance():
    x=planted('role',R=4,Q=4)
    rows,xx=row_grid(x)
    xx[0,0]+=1;xx[1,0]-=1
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(4)))
    out=A.all_valid_fit(c,raw_features=xx)
    assert out['within_prompt_ss']==pytest.approx(2)
    assert out['residual_ss']==pytest.approx(2)


def good_gate(n=24000,valid=23990):
    return {'passed':True,'gates':{
        'G1':dict(n_rows=n,n_state_rows=n,n_quote_rows=n,**{'pass':True}),
        'G2':{'pass':True},
        'G3':dict(n_rows=40,identical_same_batches=40,**{'pass':True}),
        'G4':dict(n_invalid=n-valid,speech_quote_rows_unequal=0,
                   cells={f'L{k}':dict(n=valid,min_cos=.99,median_cos=.999,median_rel_norm=.001) for k in (9,17,23)},
                   quotes={f'L{k}':dict(n=10,max_rel_norm=.01,median_rel_norm=.001) for k in (8,16,24)},**{'pass':True}),
        'G5':dict(mismatch={},**{'pass':True})}}


def test_quote_reconstruction_gate_evidence_must_cover_current_quote_rows():
    gate=good_gate()
    A.validate_gate_evidence(gate,24000,23990,n_quote_other=10)
    gate['gates']['G4']['quotes']['L16']['max_rel_norm']=.06
    with pytest.raises(A.AnalysisError,match='quote'):
        A.validate_gate_evidence(gate,24000,23990,n_quote_other=10)


def test_primary_resampling_artifacts_keep_actual_halves_and_draws(tmp_path):
    x=planted('role',R=6,Q=8)
    result=A.d1(x,n_boot=5,min_questions=7,artifacts_dir=tmp_path)
    assert len(np.load(tmp_path/'D1_bootstrap.npy'))==5
    result=A.d3(x,n_boot=3,n_perm=5,n_halves=3,artifacts_dir=tmp_path)
    z=np.load(tmp_path/'D3_resampling.npz')
    assert z['null'].shape==(5,)
    assert z['bootstrap'].shape==(3,)
    assert z['question_halves'].shape==(3,8)
    assert z['wording_halves'].shape==(3,5)


def test_d2_end_to_end_keeps_matched_readers_and_saves_all_folds(tmp_path):
    x=planted('role',R=4,Q=8)
    rows,xx=row_grid(x)
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8)))
    # The hidden activation carries role; text, embedding and state0 carry none.
    result=A.d2(c,rows,np.ones_like(xx),np.ones((len(rows),3)),n_boot=5,n_perm=2,n_splits=4,out_dir=tmp_path)
    assert len(result['formats'])==4
    assert result['p']==max(v['p'] for v in result['formats'].values())
    for value in result['formats'].values():
        assert value['accuracy']['point']==pytest.approx(1)
        assert {'word','char','sentence','hidden_state0','length','stop_reason','question_id','format_id'}==set(value['baselines'])
    inputs=json.loads((tmp_path/'D2_inputs.json').read_text())
    assert len(inputs['folds']['0'])==4
    assert len(inputs['role_draws'])==5
    assert (tmp_path/'D2_reader_cells_format0.npz').is_file()


def test_holm_family_verdict_waits_for_explicit_alpha():
    raw={'D1':{'p':.001,'raw_rule_met':True},'D2':{'p':.002,'raw_rule_met':True},'D3':{'p':.003,'raw_rule_met':True}}
    pending=A.decision_report(raw,confirmatory=True)
    assert pending['family_status']=='pending_owner_alpha'
    assert all(v['family_decision'] is None for v in pending['decisions'].values())
    assert pending['decisions']['D2']['holm_p']==pytest.approx(.004)
    chosen=A.decision_report(raw,confirmatory=True,alpha=.01)
    assert all(v['family_decision'] for v in chosen['decisions'].values())
    smoke=A.decision_report(raw,confirmatory=False,alpha=.01)
    assert all(v['family_decision'] is None for v in smoke['decisions'].values())


def test_phase_resume_rejects_changed_saved_metric(tmp_path):
    A.phase(tmp_path,'sample',lambda:{'score':.25})
    path=tmp_path/'sample.json'
    path.write_text(json.dumps({'score':.99}))
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.phase(tmp_path,'sample',lambda:pytest.fail('must not recompute a corrupt cache'))


def test_phase_resume_rejects_missing_integrity_sidecar(tmp_path):
    A.phase(tmp_path,'sample',lambda:{'score':.25})
    sidecar=tmp_path/'sample.json.integrity.json'
    assert sidecar.exists()
    sidecar.unlink()
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.phase(tmp_path,'sample',lambda:pytest.fail('must not reuse an unsealed cache'))


def test_null_resume_rejects_changed_permutation_values(tmp_path):
    rows,xx=row_grid(planted('role',R=4,Q=8))
    p=A.flatten_cells(A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8))))
    folds=A.make_splits(p,4,n=2)[0]
    path=tmp_path/'null.json'
    A.decode_null(p['X'],p,folds,n=2,checkpoint=path)
    saved=json.loads(path.read_text());saved['values'][0]=.999
    path.write_text(json.dumps(saved))
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.decode_null(p['X'],p,folds,n=2,checkpoint=path)


@pytest.mark.parametrize('artifact',['D2_inputs.json','D2_format0.json','D2_reader_cells_format0.npz','D2_null_format0.json'])
def test_d2_resume_rejects_corrupt_artifacts(tmp_path,artifact):
    x=planted('role',R=4,Q=8)
    rows,xx=row_grid(x)
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8)))
    args=(c,rows,np.ones_like(xx),np.ones((len(rows),3)))
    A.d2(*args,n_boot=3,n_perm=1,n_splits=2,out_dir=tmp_path)
    path=tmp_path/artifact
    with path.open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.d2(*args,n_boot=3,n_perm=1,n_splits=2,out_dir=tmp_path)


def test_phase_resume_checks_its_resampling_dependencies(tmp_path):
    dependency=tmp_path/'D1_bootstrap.npy'
    def compute():
        np.save(dependency,np.array([1.,2.]))
        return {'score':.5}
    A.phase(tmp_path,'D1',compute)
    np.save(dependency,np.array([100.,200.]))
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.phase(tmp_path,'D1',lambda:pytest.fail('must not reuse changed draws'))


def test_heavy_analysis_refuses_mac_before_creating_outputs(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(A.platform,'system',lambda:'Darwin')
    with pytest.raises(SystemExit) as stopped:
        A.main(['--run','unused','--out',str(tmp_path/'forbidden'),'--execute-on-pod','--raw-read-note','unused'])
    assert stopped.value.code==2
    assert 'Linux pod' in capsys.readouterr().err
    assert not (tmp_path/'forbidden').exists()


def test_phase_resume_reuses_valid_result_without_recomputing(tmp_path):
    assert A.phase(tmp_path,'sample',lambda:{'score':.25})=={'score':.25}
    assert A.phase(tmp_path,'sample',lambda:pytest.fail('must reuse intact cache'))=={'score':.25}


def test_d2_resume_reuses_all_sealed_formats(tmp_path,monkeypatch):
    rows,xx=row_grid(planted('role',R=4,Q=8))
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8)))
    args=(c,rows,np.ones_like(xx),np.ones((len(rows),3)))
    result=A.d2(*args,n_boot=3,n_perm=1,n_splits=2,out_dir=tmp_path)
    monkeypatch.setattr(A,'decode_report',lambda *a,**k:pytest.fail('must reuse intact formats'))
    assert A.d2(*args,n_boot=3,n_perm=1,n_splits=2,out_dir=tmp_path)==result


def test_sentence_reader_matches_e3_normalisation_and_float32_without_download(tmp_path,monkeypatch):
    import types
    seen={}
    class FakeSentenceTransformer:
        def __init__(self,name,**kwargs):
            seen['model']=name;seen['init']=kwargs
        def encode(self,texts,**kwargs):
            seen['texts']=texts;seen['encode']=kwargs
            return np.tile(np.array([.6,.8],dtype=np.float64),(len(texts),1))
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    run={'rows':[{'reply':'test reply','valid':True},{'reply':'','valid':False}]}
    got=A.embedding_cache(run,tmp_path)
    assert seen['init']['local_files_only'] is True
    assert seen['init']['revision']==A.EMBED_MODEL[1]
    assert seen['encode'].get('normalize_embeddings') is True
    assert got.dtype==np.float32
    metadata=json.loads((tmp_path/'sentence_embeddings.json').read_text())
    assert metadata['normalize_embeddings'] is True
    assert metadata['dtype']=='float32'
    # Reuse is local and retains the same exact encoded vectors.
    np.testing.assert_array_equal(got,A.embedding_cache(run,tmp_path))


def test_generation_rows_hash_is_historical_while_analysis_helper_can_change():
    original='b1e4c29bba60a49b785b4847c818ffff623dd5e96253b05e03df8df2a9eb0c4a'
    current='ba1e1e0321768bae59b507e36f37f3a94bccf4a8a8ef716f40d1c2b374190e16'
    generation={'gen_extract.py':'generation-script-hash','rows.py':original}
    live={'gen_extract.py':'generation-script-hash','rows.py':current,'gates.py':'gate-code-hash'}
    snapshot=generation.copy()
    record=A.validate_generation_code(generation,live)
    assert generation==snapshot
    assert record['generation_code']['rows.py']==original
    assert record['current_analysis_code']['rows.py']==current
    assert record['current_analysis_code']['gates.py']=='gate-code-hash'


@pytest.mark.parametrize('bad_hash',['unknown-generation-hash','ba1e1e0321768bae59b507e36f37f3a94bccf4a8a8ef716f40d1c2b374190e16'])
def test_unknown_or_current_rows_hash_cannot_replace_approved_generation_hash(bad_hash):
    with pytest.raises(A.AnalysisError,match='generation|Generation'):
        A.validate_generation_code({'gen_extract.py':'same','rows.py':bad_hash},
                                   {'gen_extract.py':'same','rows.py':bad_hash,'gates.py':'gate'})


def test_current_generator_must_still_match_historical_run():
    original='b1e4c29bba60a49b785b4847c818ffff623dd5e96253b05e03df8df2a9eb0c4a'
    with pytest.raises(A.AnalysisError,match='gen_extract.py'):
        A.validate_generation_code({'gen_extract.py':'old','rows.py':original},
                                   {'gen_extract.py':'changed','rows.py':'analysis-only','gates.py':'gate'})


def test_recorded_gate_script_must_still_match_current_bytes():
    original='b1e4c29bba60a49b785b4847c818ffff623dd5e96253b05e03df8df2a9eb0c4a'
    with pytest.raises(A.AnalysisError,match='gates.py'):
        A.validate_generation_code({'gen_extract.py':'same','rows.py':original,'gates.py':'old-gate'},
                                   {'gen_extract.py':'same','rows.py':'analysis-only','gates.py':'changed-gate'})


def test_analysis_provenance_keeps_historical_and_current_helper_hashes(tmp_path):
    import types
    G=A.sibling('gen_extract')
    generation={'gen_extract.py':G.sha256_file(HERE/'gen_extract.py'),'rows.py':A.GENERATION_ROWS_SHA256}
    note=tmp_path/'read-note.md';note.write_text('Synthetic provenance fixture; no real data.')
    run=dict(path=tmp_path,manifest=dict(fingerprint='synthetic',code=generation,tokenizer_files={}),
             verified=dict(raw_read_note=str(note),raw_read_sha256=G.sha256_file(note)))
    args=types.SimpleNamespace(n_boot=2000,n_perm=9999,n_splits=50,n_halves=50)
    result=A.provenance(run,args)
    current=G.sha256_file(Path(G.R.__file__))
    assert result['code_provenance']['generation_code']['rows.py']==A.GENERATION_ROWS_SHA256
    assert result['code_provenance']['current_analysis_code']['rows.py']==current
    assert result['source_hashes'][str(Path(G.R.__file__))]==current
    assert generation['rows.py']==A.GENERATION_ROWS_SHA256


def test_fingerprint_accepts_versions_extended_by_generation_finalizer():
    G=A.sibling('gen_extract')
    parts=dict(model='base',revision='frozen',versions=dict(python='3.12.3',torch='2.8.0+cu128',transformers='4.47.1',tokenizers='0.21.0'))
    manifest=dict(parts,fingerprint=G.R.fingerprint(parts),versions=dict(parts['versions'],cuda='12.8',numpy='2.1.2'))
    recorded=dict(parts,template_text={'descriptive':'not a fingerprint field'})
    assert A.validate_fingerprint_parts(recorded,manifest)==parts
    assert 'template_text' in recorded  # Validation never rewrites the input record.


@pytest.mark.parametrize('change',['historical_version','model','fingerprint'])
def test_fingerprint_still_rejects_changed_historical_fields(change):
    G=A.sibling('gen_extract')
    parts=dict(model='base',versions=dict(torch='2.8.0+cu128'))
    manifest=dict(parts,fingerprint=G.R.fingerprint(parts),versions=dict(parts['versions'],cuda='12.8'))
    if change=='historical_version':manifest['versions']['torch']='2.13.0+cu129'
    elif change=='model':manifest['model']='different'
    else:manifest['fingerprint']='different'
    with pytest.raises(A.AnalysisError,match='fingerprint|Fingerprint'):
        A.validate_fingerprint_parts(parts,manifest)


def small_null_inputs(seed=101):
    rng=np.random.default_rng(seed)
    x=planted('role',R=4,Q=8)+rng.normal(0,1.7,size=(4,4,8,5,7))
    rows,xx=row_grid(x,nrep=1)
    p=A.flatten_cells(A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8))))
    return p,A.make_splits(p,4,n=3,seed=17)[0]


@pytest.mark.parametrize('cache_scalers',[False,True])
def test_parallel_cached_null_equals_serial_every_draw_and_p(tmp_path,cache_scalers):
    p,folds=small_null_inputs()
    serial=A.decode_null(p['X'],p,folds,n=7,seed=79,blas_threads=1)
    parallel=A.decode_null(p['X'],p,folds,n=7,seed=79,workers=2,blas_threads=1,cache_scalers=cache_scalers,
                           checkpoint=tmp_path/'parallel.json')
    assert parallel['observed']==serial['observed']
    assert parallel['null']==serial['null']
    assert parallel['p']==serial['p']
    saved=json.loads((tmp_path/'parallel.json').read_text())
    assert saved['execution']==dict(workers=2,blas_threads=1,cache_scalers=cache_scalers,backend='threads')


def test_cached_scaler_predictions_equal_fresh_e3_fits_and_are_read_only(tmp_path):
    p,folds=small_null_inputs()
    original=p['X'].copy()
    prepared=A.prepare_null_folds(p['X'],p,folds,tmp_path)
    for fold,(tr,te) in zip(prepared,folds):
        expected=A.fit_predict(p['X'][tr],p['y'][tr],p['X'][te])
        actual=A.fit_predict_scaled(fold['Xtr'],p['y'][tr],fold['Xte'])
        np.testing.assert_array_equal(actual,expected)
        assert not fold['Xtr'].flags.writeable
        assert not fold['Xte'].flags.writeable
        assert isinstance(fold['Xtr'],np.memmap)
    np.testing.assert_array_equal(p['X'],original)


def test_cached_scaler_depends_only_on_training_x(tmp_path):
    p,folds=small_null_inputs()
    first=A.prepare_null_folds(p['X'],p,folds[:1],tmp_path/'first')[0]
    changed=p['X'].copy();changed[folds[0][1]]+=10000
    other=dict(p,y=A.block_permute(p['y'],p['q'],np.random.default_rng(24)))
    second=A.prepare_null_folds(changed,other,folds[:1],tmp_path/'second')[0]
    np.testing.assert_array_equal(first['Xtr'],second['Xtr'])
    assert not np.array_equal(first['Xte'],second['Xte'])


@pytest.mark.parametrize('leak',['same_row','question','wording','format'])
def test_parallel_null_rejects_leaking_folds_before_fitting(leak):
    p,folds=small_null_inputs()
    tr,te=folds[0]
    if leak=='same_row':
        tr=np.r_[tr,te[:1]]
    else:
        p=dict(p)
        name={'question':'q','wording':'w','format':'f'}[leak]
        p[name]=p[name].copy();p[name][te[0]]=p[name][tr[0]]
    with pytest.raises(A.AnalysisError,match='overlap|held|leak|disjoint'):
        A.decode_null(p['X'],p,[(tr,te)],n=2,workers=2,blas_threads=1,cache_scalers=True)


def test_parallel_null_interruption_resumes_exact_ordered_seed_prefix(tmp_path,monkeypatch):
    p,folds=small_null_inputs()
    full=A.decode_null(p['X'],p,folds,n=6,seed=13,workers=2,blas_threads=1,cache_scalers=True)
    path=tmp_path/'interrupted.json'
    real_save=A.save_checked_json
    def interrupt_after_two(path,value,*args,**kwargs):
        real_save(path,value,*args,**kwargs)
        if len(value.get('values',[]))==2:
            raise KeyboardInterrupt('synthetic interruption after durable prefix')
    monkeypatch.setattr(A,'save_checked_json',interrupt_after_two)
    with pytest.raises(KeyboardInterrupt):
        A.decode_null(p['X'],p,folds,n=6,seed=13,workers=2,blas_threads=1,cache_scalers=True,checkpoint=path)
    assert json.loads(path.read_text())['values']==full['null'][:2]
    monkeypatch.setattr(A,'save_checked_json',real_save)
    resumed=A.decode_null(p['X'],p,folds,n=6,seed=13,workers=2,blas_threads=1,cache_scalers=True,checkpoint=path)
    assert resumed==full


def test_parallel_null_corrupt_resume_is_refused(tmp_path):
    p,folds=small_null_inputs()
    path=tmp_path/'corrupt-parallel.json'
    A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,cache_scalers=True,checkpoint=path)
    path.write_text(path.read_text()+' ')
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,cache_scalers=True,checkpoint=path)


def test_parallel_null_requires_explicit_native_thread_budget():
    p,folds=small_null_inputs()
    with pytest.raises(A.AnalysisError,match='thread|resource'):
        A.decode_null(p['X'],p,folds,n=1,workers=2)


def test_parallel_null_still_refits_every_frozen_classifier(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    p,folds=small_null_inputs()
    models=[]
    real_fit=LogisticRegression.fit
    def record_fit(self,X,y,*args,**kwargs):
        models.append(self)
        assert self.C==1.0 and self.class_weight=='balanced' and self.max_iter==2000
        return real_fit(self,X,y,*args,**kwargs)
    monkeypatch.setattr(LogisticRegression,'fit',record_fit)
    A.decode_null(p['X'],p,folds,n=4,workers=2,blas_threads=1,cache_scalers=True)
    assert len(models)==(4+1)*len(folds)
    assert len({id(model) for model in models})==len(models)


def test_parallel_resources_are_recorded_for_provenance(monkeypatch):
    import types
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','7')
    args=types.SimpleNamespace(null_workers=2,null_blas_threads=3,cache_null_scalers=True)
    profile=A.null_resource_provenance(args)
    assert profile['execution']==dict(workers=2,blas_threads=3,cache_scalers=True,backend='threads')
    assert profile['thread_environment']['OPENBLAS_NUM_THREADS']=='7'
    assert profile['max_in_flight_draws']==2
    assert profile['scaler_cache_scope']=='temporary_per_call_rebuilt_on_resume'


def test_parallel_null_resource_changes_cannot_reuse_a_checkpoint(tmp_path):
    p,folds=small_null_inputs()
    path=tmp_path/'resources.json'
    A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,cache_scalers=True,checkpoint=path)
    with pytest.raises(A.AnalysisError,match='fingerprint|resource'):
        A.decode_null(p['X'],p,folds,n=2,workers=1,blas_threads=1,cache_scalers=True,checkpoint=path)


def test_d2_passes_and_records_optional_parallel_resources(tmp_path):
    x=planted('role',R=4,Q=8)
    rows,xx=row_grid(x)
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8)))
    result=A.d2(c,rows,np.ones_like(xx),np.ones((len(rows),3)),n_boot=3,n_perm=1,n_splits=2,
                out_dir=tmp_path,null_workers=2,null_blas_threads=1,cache_null_scalers=True)
    expected=dict(workers=2,blas_threads=1,cache_scalers=True,backend='threads')
    assert result['null_execution']==expected
    assert json.loads((tmp_path/'D2_inputs.json').read_text())['null_execution']==expected
    assert all(value['null_execution']==expected for value in result['formats'].values())


def test_null_checks_resource_matched_observed_score_before_any_draw(monkeypatch):
    p,folds=small_null_inputs()
    observed=A.decode(p['X'],p,folds)['accuracy']
    monkeypatch.setattr(A,'block_permute',lambda *a,**k:pytest.fail('must stop before any null permutation'))
    with pytest.raises(A.AnalysisError,match='observed|primary'):
        A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,cache_scalers=True,
                       expected_observed=observed-.01)


def test_streamed_feature_hash_matches_original_contiguous_fingerprint():
    import hashlib
    x=np.arange(15000,dtype=np.float64).reshape(3000,5)[:,::-1]
    assert A.feature_sha256(x)==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@pytest.mark.parametrize('cache_scalers',[False,True])
def test_process_null_matches_serial_from_importlib_module(tmp_path,cache_scalers):
    p,folds=small_null_inputs()
    serial=A.decode_null(p['X'],p,folds,n=5,seed=73,blas_threads=1)
    process=A.decode_null(p['X'],p,folds,n=5,seed=73,workers=2,blas_threads=1,
                          cache_scalers=cache_scalers,backend='processes',checkpoint=tmp_path/'process.json')
    assert process==serial
    assert json.loads((tmp_path/'process.json').read_text())['execution']['backend']=='processes'


@pytest.mark.parametrize('cache_scalers',[False,True])
def test_process_pool_memmaps_inner_limits_and_warning_errors(tmp_path,cache_scalers):
    import os
    from sklearn.exceptions import ConvergenceWarning
    p,folds=small_null_inputs()
    prepared=A.prepare_null_folds(p['X'],p,folds,tmp_path/'scaled') if cache_scalers else None
    pool,worker=A._new_process_null_pool(p['X'],p,folds,prepared,tmp_path/'pool',17,2,1)
    try:
        state=f"__import__('sys').modules[{worker.__module__!r}]._PROCESS_NULL_STATE"
        expression=("{'pid': __import__('os').getpid(), 'readonly': "
                    f"all(not f[k].flags.writeable and type(f[k]).__name__=='memmap' for f in {state}['prepared'] for k in ('Xtr','Xte'))"
                    if cache_scalers else
                    "{'pid': __import__('os').getpid(), 'readonly': "
                    f"not {state}['X'].flags.writeable and type({state}['X']).__name__=='memmap'")
        expression+=", 'pools': __import__('threadpoolctl').threadpool_info(), 'omp': __import__('os').environ['OMP_NUM_THREADS']}"
        result=pool.submit(eval,expression).result(timeout=30)
        assert result['pid']!=os.getpid()
        assert result['readonly'] and result['omp']=='1'
        assert all(info['num_threads']==1 for info in result['pools'] if info['user_api'] in ('blas','openmp'))
        with pytest.raises(ConvergenceWarning):
            pool.submit(exec,"import warnings; from sklearn.exceptions import ConvergenceWarning; warnings.warn('synthetic convergence probe', ConvergenceWarning)").result(timeout=30)
    finally:
        pool.shutdown(wait=True)


def test_process_null_interrupted_prefix_resumes_identically(tmp_path,monkeypatch):
    p,folds=small_null_inputs()
    full=A.decode_null(p['X'],p,folds[:2],n=4,seed=23,workers=2,blas_threads=1,cache_scalers=True,backend='processes')
    path=tmp_path/'process-prefix.json';real_save=A.save_checked_json
    def interrupt(path,value,*args,**kwargs):
        real_save(path,value,*args,**kwargs)
        if len(value.get('values',[]))==2:
            raise KeyboardInterrupt('synthetic process prefix interruption')
    monkeypatch.setattr(A,'save_checked_json',interrupt)
    with pytest.raises(KeyboardInterrupt):
        A.decode_null(p['X'],p,folds[:2],n=4,seed=23,workers=2,blas_threads=1,cache_scalers=True,
                      backend='processes',checkpoint=path)
    assert json.loads(path.read_text())['values']==full['null'][:2]
    monkeypatch.setattr(A,'save_checked_json',real_save)
    assert A.decode_null(p['X'],p,folds[:2],n=4,seed=23,workers=2,blas_threads=1,cache_scalers=True,
                         backend='processes',checkpoint=path)==full
    assert not list(tmp_path.glob('.podb-null-folds-*'))


def test_process_corruption_and_backend_resource_mismatch_refuse_resume(tmp_path,monkeypatch):
    p,folds=small_null_inputs()
    path=tmp_path/'process-corrupt.json'
    A.decode_null(p['X'],p,folds[:2],n=2,workers=2,blas_threads=1,cache_scalers=True,
                  backend='processes',checkpoint=path)
    monkeypatch.setattr(A,'_new_process_null_pool',lambda *a,**k:pytest.fail('must reject cache before starting children'))
    with pytest.raises(A.AnalysisError,match='fingerprint|resource'):
        A.decode_null(p['X'],p,folds[:2],n=2,workers=2,blas_threads=1,cache_scalers=True,checkpoint=path)
    path.write_text(path.read_text()+' ')
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.decode_null(p['X'],p,folds[:2],n=2,workers=2,blas_threads=1,cache_scalers=True,
                      backend='processes',checkpoint=path)


def test_process_backend_resource_provenance_and_explicit_limits():
    import types
    profile=A.null_resource_provenance(types.SimpleNamespace(null_workers=2,null_blas_threads=1,
                                       cache_null_scalers=True,null_backend='processes'))
    assert profile['execution']['backend']=='processes'
    assert profile['process_implementation']=='joblib.loky'
    assert profile['inner_limit_scope']=='BLAS_and_OpenMP_in_each_child'
    with pytest.raises(A.AnalysisError,match='thread|resource'):
        A.null_execution_settings(workers=1,backend='processes')


def test_d2_process_backend_is_wired_and_recorded(tmp_path):
    x=planted('role',R=4,Q=8);rows,xx=row_grid(x)
    c=A.prompt_cells(rows,xx,[f'r{i}' for i in range(4)],[f'f{i}' for i in range(4)],list(range(8)))
    result=A.d2(c,rows,np.ones_like(xx),np.ones((len(rows),3)),n_boot=3,n_perm=1,n_splits=2,
                out_dir=tmp_path,null_workers=2,null_blas_threads=1,cache_null_scalers=True,null_backend='processes')
    assert result['null_execution']['backend']=='processes'
    assert all(value['null_execution']['backend']=='processes' for value in result['formats'].values())
    assert json.loads((tmp_path/'D2_inputs.json').read_text())['null_execution']['backend']=='processes'


def test_process_backend_cli_option_is_exposed():
    import subprocess
    result=subprocess.run([sys.executable,str(HERE/'analysis.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--null-backend {threads,processes,batches}' in result.stdout


def test_fast_d1_uses_existing_draws_and_keeps_reference_decisions():
    rng=np.random.default_rng(95)
    x=rng.normal(size=(5,4,8,5,7))
    draws=A.crossed_draws(5,8,30,17)
    original=np.array([A.variance_split(x[r][:,:,q],False)['role_minus_format'] for r,q in draws])
    result=A.d1(x,n_boot=30,seed=17,min_questions=2)
    np.testing.assert_allclose([result['difference']['lo'],result['difference']['hi']],
                               np.percentile(original,[2.5,97.5]),atol=2e-13,rtol=2e-12)
    assert result['p']==float(np.mean(original<=0))
    assert result['raw_rule_met']==bool(np.percentile(original,2.5)>0)
    assert result['bootstrap_execution']['n_draws']==30


def test_cuda_execution_requires_exact_transforms_and_one_inner_thread():
    for cache,threads in [(False,1),(True,2)]:
        with pytest.raises(A.AnalysisError,match='CUDA'):
            A.null_execution_settings(2,threads,cache,'processes','cuda')
    cfg=A.null_execution_settings(2,1,True,'processes','cuda')
    assert cfg['device']=='cuda' and cfg['precision']=='float64'


def test_cuda_wiring_preserves_draws_and_verifies_observed_predictions(tmp_path,monkeypatch):
    p,folds=small_null_inputs();helper=A.sibling('analysis_gpu');calls=[]
    original=p['X'].copy()
    monkeypatch.setattr(helper,'prepare_cuda_folds',lambda prepared:list(prepared))
    def checked_fit(x,y,z):
        calls.append(len(y));return A._fit_predict_scaled(x,y,z)
    monkeypatch.setattr(helper,'fit_predict_scaled_array_api',checked_fit)
    monkeypatch.setattr(helper,'release_cuda_folds',lambda prepared:prepared.clear())
    expected=A.decode_null(p['X'],p,folds,n=3,seed=97,blas_threads=1)
    actual=A.decode_null(p['X'],p,folds,n=3,seed=97,workers=2,blas_threads=1,
                         cache_scalers=True,backend='threads',device='cuda',checkpoint=tmp_path/'cuda.json')
    assert actual==expected
    assert len(calls)==(3+1)*len(folds)
    assert json.loads((tmp_path/'cuda.json').read_text())['execution']['device']=='cuda'
    np.testing.assert_array_equal(p['X'],original)


def test_cuda_observed_mismatch_stops_before_null_draws(monkeypatch):
    p,folds=small_null_inputs();helper=A.sibling('analysis_gpu')
    monkeypatch.setattr(helper,'prepare_cuda_folds',lambda prepared:list(prepared))
    monkeypatch.setattr(helper,'fit_predict_scaled_array_api',lambda x,y,z:np.full(len(z),-1))
    monkeypatch.setattr(A,'block_permute',lambda *a,**k:pytest.fail('No null draw may start after mismatch'))
    with pytest.raises(A.AnalysisError,match='CUDA observed predictions'):
        A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,cache_scalers=True,device='cuda')


def test_true_batch_null_matches_every_original_draw_and_resumes(tmp_path):
    p,folds=small_null_inputs()
    reference=A.decode_null(p['X'],p,folds,n=5,seed=97,blas_threads=1)
    path=tmp_path/'null.json'
    result=A.decode_null(p['X'],p,folds,n=5,seed=97,workers=3,blas_threads=1,
                         cache_scalers=True,backend='batches',device='cpu',checkpoint=path)
    assert result==reference
    assert A.decode_null(p['X'],p,folds,n=5,seed=97,workers=3,blas_threads=1,
                         cache_scalers=True,backend='batches',device='cpu',checkpoint=path)==reference


def test_true_batch_refuses_changed_execution_or_corrupt_prefix(tmp_path):
    p,folds=small_null_inputs();path=tmp_path/'null.json'
    A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,
                  cache_scalers=True,backend='batches',device='cpu',checkpoint=path)
    with pytest.raises(A.AnalysisError,match='fingerprint|resources'):
        A.decode_null(p['X'],p,folds,n=2,workers=3,blas_threads=1,
                      cache_scalers=True,backend='batches',device='cpu',checkpoint=path)
    path.write_text(path.read_text()+' ')
    with pytest.raises(A.AnalysisError,match='integrity|checksum'):
        A.decode_null(p['X'],p,folds,n=2,workers=2,blas_threads=1,
                      cache_scalers=True,backend='batches',device='cpu',checkpoint=path)


def test_true_batch_requires_frozen_float64_resource_bounds():
    for workers,threads,cache in [(513,1,True),(4,2,True),(4,1,False)]:
        with pytest.raises(A.AnalysisError):A.null_execution_settings(workers,threads,cache,'batches','cuda')


def test_true_batch_interruption_keeps_exact_completed_draw_prefix(tmp_path,monkeypatch):
    p,folds=small_null_inputs();path=tmp_path/'null.json'
    reference=A.decode_null(p['X'],p,folds,n=5,seed=43,blas_threads=1)
    original=A.save_checked_json
    def interrupted(file,value,*args,**kwargs):
        result=original(file,value,*args,**kwargs)
        if Path(file)==path and len(value.get('values',[]))==2:raise KeyboardInterrupt('bounded interruption')
        return result
    monkeypatch.setattr(A,'save_checked_json',interrupted)
    with pytest.raises(KeyboardInterrupt):
        A.decode_null(p['X'],p,folds,n=5,seed=43,workers=3,blas_threads=1,
                      cache_scalers=True,backend='batches',device='cpu',checkpoint=path)
    monkeypatch.setattr(A,'save_checked_json',original)
    assert A.checked_json(path,dependencies=())['values']==reference['null'][:2]
    assert A.decode_null(p['X'],p,folds,n=5,seed=43,workers=3,blas_threads=1,
                         cache_scalers=True,backend='batches',device='cpu',checkpoint=path)==reference


def test_cuda_runtime_is_recorded_without_changing_cpu_defaults(monkeypatch):
    from types import SimpleNamespace
    helper=A.sibling('analysis_gpu')
    monkeypatch.setattr(helper,'runtime_info',lambda:dict(device='synthetic_gpu',precision='float64'))
    report=A.null_resource_provenance(SimpleNamespace(null_workers=2,null_blas_threads=1,
                cache_null_scalers=True,null_backend='processes',null_device='cuda'))
    assert report['gpu']['device']=='synthetic_gpu'
    assert A.null_execution_settings()==dict(workers=1,blas_threads=None,cache_scalers=False,backend='threads')
