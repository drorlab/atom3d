for REP in $(seq 1 2 3); do
        for CUTOFF in 08 09 10 11 12 13 14;  do
		for BATCHSIZE in 1 2; do
			FORMAT=LMDB-noH
			SB_FILE=msp_cutoff${CUTOFF}_bs${BATCHSIZE}_${FORMAT}_rep${REP}.sb
			cp template-msp.sb ${SB_FILE}
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

