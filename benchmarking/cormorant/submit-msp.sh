for REP in $(seq 1 2 3); do
        for CUTOFF in 06 07 08 09 10 12 15;  do
		for BATCHSIZE in 1 2 4 8; do
			for FORMAT in NPZ LMDB; do
				SB_FILE=msp_cutoff${CUTOFF}_bs${BATCHSIZE}_${FORMAT}_rep${REP}.sb
				cp template-msp.sb ${SB_FILE}
                                sed -i "s/FORMAT/${FORMAT}/g" ${SB_FILE}
				sed -i "s/BATCHSIZE/${BATCHSIZE}/g" ${SB_FILE}
                		sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
				sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
				sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
				sbatch ${SB_FILE}
			done
		done
        done
done

