# -- Source directory paths --

RAWDIR=../../data/ligand_efficacy_prediction/split


for MAXNUMAT in 500 480 520; do

	for CUTOFF in 50 55 60; do
		
		NPZDIR=../../data/ligand_efficacy_prediction/split/npz_cutoff${CUTOFF}_maxnumat${MAXNUMAT}

		mkdir $NPZDIR
		mkdir $NPZDIR/lep
		
		CUTOFFNM=$( printf %.1f "$((10**1 * $CUTOFF/10))e-1" )
		echo "Processing with cut-off $CUTOFFNM nm and max. $MAXNUMAT atoms, dropping hydrogen atoms."
		python ../../atom3d/datasets/lep/convert_lep_to_npz.py $RAWDIR $NPZDIR/lep --cutoff $CUTOFFNM --maxnumat $MAXNUMAT --drop_h

	done
done


for MAXNUMAT in 500 480 520; do

        for CUTOFF in 40 45 50; do

                NPZDIR=../../data/ligand_efficacy_prediction/split/npz_cutoff${CUTOFF}_maxnumat${MAXNUMAT}

                mkdir $NPZDIR
                mkdir $NPZDIR/lep

                CUTOFFNM=$( printf %.1f "$((10**1 * $CUTOFF/10))e-1" )
                echo "Processing with cut-off $CUTOFFNM nm and max. $MAXNUMAT atoms, including hydrogen atoms."
                python ../../atom3d/datasets/lep/convert_lep_to_npz.py $RAWDIR $NPZDIR/lep --cutoff $CUTOFFNM --maxnumat $MAXNUMAT

        done
done

