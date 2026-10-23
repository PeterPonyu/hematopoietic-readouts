#!/usr/bin/env python3
"""One-time local export; accepts a repository root, never fetches or trains."""
import argparse, hashlib, json, shutil
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--cohorts',nargs='+',choices=['irradiation','ucb','dapp1','sleep','marrow','dapp1_spliced'],default=['irradiation','ucb','dapp1','sleep','marrow','dapp1_spliced']);a=p.parse_args()
    root=a.repository_root.resolve();out=a.output_root.resolve(); old=root/'results/application_reanalysis_20260921';new=root/'results/recovered_integration_20261001';manifest=[];maps=[]
    if (out/'support_manifest.json').exists():manifest=json.loads((out/'support_manifest.json').read_text())['files']
    if (out/'acquisition_manifest.json').exists():maps=json.loads((out/'acquisition_manifest.json').read_text())['sources']
    def register(entry):
        manifest[:]=[row for row in manifest if row['path']!=entry['path']]
        manifest.append(entry)
    def cp(src,relative,role):
        dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        register({'path':str(relative),'source':str(src.relative_to(root)),'sha256':sha(dst),'bytes':dst.stat().st_size,'role':role})
    for c in a.cohorts:
        source=old/c; dest=out/'inputs/frozen'/c;dest.mkdir(parents=True,exist_ok=True)
        for name in ['cells.csv.gz','input.npz','protocol.json','marker_log1p.csv.gz']:
            cp(source/name,Path('inputs/frozen')/c/name,'frozen cell axis, annotation, coordinate or expression reference')
        for name in ['qc_cells.csv.gz','loom_layer_identity.json','geo_identity.json']:
            if (source/name).exists():cp(source/name,Path('inputs/frozen')/c/name,'preparation provenance')
        proto=json.loads((source/'protocol.json').read_text()); exp=pd.read_csv(source/'marker_log1p.csv.gz',index_col=0)
        with h5py.File(source/'prepared.h5ad') as h:
            decode=lambda x:np.array([v.decode() if isinstance(v,bytes) else str(v) for v in x],dtype=str)
            cluster_order=decode(h['obs/leiden/categories'][:]).tolist()
            genes=decode(h['var'][h['var'].attrs.get('_index','_index')][:]); ids=decode(h['obs'][h['obs'].attrs.get('_index','_index')][:]);g=h['layers/counts'];shape=tuple(g.attrs['shape']);ptr=g['indptr'][:]
            select=pd.Index(genes).get_indexer(exp.columns);assert (select>=0).all();assert np.array_equal(ids,exp.index.astype(str))
            counts=np.zeros((shape[0],len(select)),np.float32);total=np.zeros(shape[0],np.int64)
            for start in range(0,shape[0],512):
                stop=min(start+512,shape[0]);lo,hi=ptr[start],ptr[stop];d=g['data'][lo:hi];assert (d>=0).all() and np.array_equal(d,np.round(d))
                block=sparse.csr_matrix((d,g['indices'][lo:hi],ptr[start:stop+1]-lo),shape=(stop-start,shape[1]))
                counts[start:stop]=block[:,select].toarray();total[start:stop]=np.asarray(block.sum(axis=1)).ravel().astype(np.int64)
            assert (total>0).all();np.savez_compressed(dest/'expression_support.npz',cell_id=ids,full_retained_genes=genes,marker_genes=np.array(exp.columns,dtype=str),marker_gene_indices=select.astype(np.int64),marker_counts=counts.astype(np.uint32),library_counts_all_retained_genes=total)
        support={'frozen_leiden_category_order':cluster_order,'cohort':c,'cell_count':len(ids),'retained_gene_count':len(genes),'marker_gene_count':len(select),'prepared_sha256':sha(source/'prepared.h5ad'),'schema':'full retained gene identities and order; integer counts for all program/marker genes; integer library size over the FULL retained gene axis','not_included':'full original local matrix; only retained compact support is included' if c=='irradiation' else 'non-marker gene count matrix; public source files are acquired separately','expression_formula':'log1p(marker_counts / (library_counts_all_retained_genes / 10000)), float32; gene z-scores use ddof=0; program = unweighted mean of available specified gene z-scores','frozen_membership':'cell_id, condition, leiden, cell_type and coordinates retained; no relabeling','historical_gap':'Dapp1 original prefilter and UMAP recipe not reconstructed; frozen cell/gene identities and coordinates supplied' if c.startswith('dapp1') else 'Author-owned historical local irradiation series; dose, animal count, source collection protocol and ethics identifiers are unavailable and are not inferred. The historical protocol accession does not establish full-series public identity.' if c=='irradiation' else None}
        save(dest/'expression_support.json',support)
        for name in ['expression_support.npz','expression_support.json']:
            f=dest/name;register({'path':str(f.relative_to(out)),'source':str((source/'prepared.h5ad').relative_to(root)),'sha256':sha(f),'bytes':f.stat().st_size,'role':'compact full-axis normalization and marker-count support'})
        if c!='dapp1_spliced':
            for r in sorted(source.glob('*/receipt.json')):
                rec=json.loads(r.read_text())
                if c=='marrow' and rec['method']!='scRL':continue
                for name in ['receipt.json','outputs.npz']:cp(r.parent/name,Path('inputs/frozen')/c/r.parent.name/name,'saved per-cell model output or binding receipt')
        for sourcefile in proto['sources']:
            basename=Path(sourcefile['path']).name;public=not basename.endswith('.h5ad')
            if public:
                src= root/'data/application_reanalysis_20260921/raw'/proto['accession']/basename
                assert src.exists() and sha(src)==sourcefile['sha256']
            maps[:]=[row for row in maps if (row['cohort'],row['file'])!=(c,basename)]
            if c=='irradiation':
                maps.append({'cohort':c,'accession':None,'historical_protocol_accession':proto['accession'],'file':basename,'sha256':sourcefile['sha256'],'public_source':False,'acquisition_url':None,'role':'author-owned historical local irradiation matrix; full series is not assigned the historical public accession','condition_mapping':'frozen d0, d2, d5, d8, d11, d14, d21 and d30 labels; experimental protocol metadata unavailable','barcode_mapping':'exact frozen cell IDs and selected count support supplied','public_identity_limit':'Earlier validation linked the day-2 subset only; this does not identify the entire local matrix as GSE278673','reuse_authority':'Author confirms ownership and authorizes computational reuse; no new animal experiments'})
            else:
                maps.append({'cohort':c,'accession':proto['accession'],'file':basename,'sha256':sourcefile['sha256'],'public_source':public,'acquisition_url':f"https://ftp.ncbi.nlm.nih.gov/geo/series/{proto['accession'][:-3]}nnn/{proto['accession']}/suppl/{proto['accession']}_RAW.tar" if public else None,'role':'raw deposited counts' if public else 'historical membership/coordinates; not required by saved replay','condition_mapping':'GSM8518011_wt -> wt; GSM8518012_ko -> ko' if c.startswith('dapp1') else 'CONTROL -> CT; SD -> SD' if c=='sleep' else 'filename population or day token; CT -> D0 for UCB','barcode_mapping':'historical regex :([ACGT]{16})x -> barcode-1 within condition; genes exact unique-name lookup' if c.startswith('dapp1') else 'source barcode with concatenated condition suffix; exact frozen IDs supplied'})
        print('exported expression',c,len(ids),len(genes),flush=True)
    for c in [c for c in ['irradiation','ucb','dapp1','sleep'] if c in a.cohorts]:
        for arm in ['PCA','LiVAE']:
            source=new/'paired_downstream'/arm/c
            for name in ['input.npz','cells.csv.gz','protocol.json','preparation_receipt.json']:
                cp(source/name,Path('inputs/matched')/arm/c/name,'matched interface')
            for r in sorted(source.glob('*/receipt.json')):
                for name in ['receipt.json','outputs.npz']:cp(r.parent/name,Path('inputs/matched')/arm/c/r.parent.name/name,'matched per-cell model output or binding receipt')
        for name in ['frozen_counts.npz','frozen_axis.npz','input.npz','representation.pt','training_history.csv','protocol.json','receipt.json','numerical_contract_probe.json']:
            cp(new/'native_representation'/c/name,Path('inputs/encoders')/c/name,'selected encoder checkpoint, selected-feature counts, history and identities')
    for name in ['module.py','utils.py']:
        cp(root/'code/LiVAE/livae'/name,Path('vendor/livae')/name,'unmodified MIT source needed to load encoder')
    cp(root/'code/LiVAE/LICENSE',Path('vendor/LiVAE-LICENSE'),'upstream MIT license')
    (out/'vendor/livae/__init__.py').write_text('"""Minimal import surface for the byte-preserved encoder module."""\n')
    for name in ['environment_manifest.json']:
        cp(old/name,Path('archival')/name,'dated environment evidence, not a tested lock')
    cp(root/'env_mirror/requirements-formal.txt',Path('archival/requirements-formal.txt'),'archival dependency template, not a tested lock')
    for acc in ['GSE280270','GSE277292','GSE280145','GSE117498']:
        for suffix in ['_family.soft.gz','_filelist.txt','_download.json']:
            src=root/'data/application_reanalysis_20260921/raw'/f'{acc}{suffix}'
            if src.exists():cp(src,Path('archival')/src.name,'cached GEO metadata or acquisition receipt; online access not retested')
    expected=root/'paper/scientific-reports-hematopoietic-readouts/source_data'
    for f in sorted(expected.rglob('*.csv')):
        relative=f.relative_to(expected);original=root/relative
        assert original.is_file(),original
        cp(original,Path('expected')/relative,'full original frozen comparison target; never used as calculation input')
    save(out/'acquisition_manifest.json',{'sources':maps,'irradiation':'author-owned historical local irradiation series; historical protocol accession preserved only as provenance, not full-series public identity; computational reuse authorized by author','online_acquisition_tested':False,'rights':'Original display and saved-output replay utilities: scoped MIT terms in reproducibility/LICENSES.txt. Upstream software notices are retained, including the MIT LiVAE module. Source data and third-party materials retain their original terms and attribution; the software license does not establish ethics/consent or relicense datasets.'})
    save(out/'support_manifest.json',{'schema_version':1,'files':manifest,'scope':'4 application cohorts (irradiation, ucb, dapp1, sleep), plus marrow scRL reward control and Dapp1 spliced expression sensitivity','count_support':'All retained gene identities, full-axis per-cell library counts and all scored-marker counts; not full non-marker expression matrix.'})
if __name__=='__main__':main()
