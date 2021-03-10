for REP in $(seq 1 3); do
        for CUTOFF in 08;  do
		for BATCHSIZE in 4; do
			FORMAT=LMDB-noH
			SB_FILE=run_cutoff${CUTOFF}_bs${BATCHSIZE}_${FORMAT}_rep${REP}.sb
			cp template-run.sb ${SB_FILE}
                        sed -i "s/FORMATDIR/LMDBDIR/g" ${SB_FILE}
                        sed -i "s/FORMAT/${FORMAT}/g" ${SB_FILE}
			sed -i "s/BATCHSIZE/${BATCHSIZE}/g" ${SB_FILE}
                	sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
			sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --drop --load/g" ${SB_FILE}
			sed -i "s/-cormorant/-cormorant-noh-rep${REP}/g" ${SB_FILE}
			sbatch ${SB_FILE}
		done
        done
done

