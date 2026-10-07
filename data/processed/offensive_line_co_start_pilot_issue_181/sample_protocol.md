# Issue 181 pilot sample freeze

Frozen on 2026-10-06 before searching for or acquiring offensive-line starter sources.
The selection target is 40 team-seasons: four two-season windows (2008–09,
2014–15, 2019–20, and 2023–24), with five programs in each window. This
provides coverage of the 2000s, 2010s, and recent seasons while allowing
adjacent-season identity matching within each selected program.

The frame is deliberately stratified into large programs and smaller programs.
Each window contributes two large and three smaller programs, selected without
replacement from the fixed ordered candidate lists in `sample_frame.csv`.
To avoid overweighting any one program, a program selected in an earlier window
is excluded from later windows. The draw uses Python's standard-library
`random.Random(181).sample`, with the window order and candidate order exactly
as recorded in `sample_frame.csv`. The resulting 40 rows are frozen in
`sample.csv`; source difficulty will not change sample membership.

This is a stratified-random pilot sample from a manually assembled candidate
frame, not a probability sample of all FBS team-seasons. The `large` and
`smaller` strata are study-design labels rather than a formal financial or
conference classification. Conference diversity is recorded after the sample
is frozen. All selected programs were treated as FBS for both seasons in their
pair when constructing the frame; transition status and historical conference
membership will be noted if a team source reveals a distinction.
