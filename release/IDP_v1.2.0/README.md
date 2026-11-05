# ProtoEss feature/model/sequence release v1.2.0

This focused archive contains feature/model workflows, frozen full-application weights for the three modelled malaria species, and the sequence resources available for the manuscript. It does not include manuscript figures, result tables, feature matrices, or general analysis scripts.

## Model scope

The current nine-input LightGBM meta-model and its route models are provided for *P. falciparum*, *P. berghei* and *P. knowlesi*. The *T. gondii* sequence set is included because Tg is a held-out species in transfer analyses; no Tg species-matched deployment model is included here. Keep all route names and configs intact. In particular, the `cross_species_with_pk_oberstaller_labels` route incorporates human-RBC Pk screen labels and must not be presented as independent prediction of that same screen.

## Sequence scope

For Pf/Pb/Pk monkey-RBC/Tg, the release contains class-separated CDS sequences for labelled genes. Pf/Pb/Pk also contain the available labelled-protein FASTA query sets. Toxoplasma protein records are translations of the released CDS sequences using nuclear genetic code 1 and are explicitly labelled as derived. The Pk A1-H.1 human-RBC assembly files are the user-supplied NCBI protein and CDS FASTAs. These are not a claim that complete proteomes/CDS for all five condition/species entries are available in this archive.

Sequence provenance and upstream terms are described in `DATA_NOTICE_upstream.md`; original database and study attribution remains applicable.
