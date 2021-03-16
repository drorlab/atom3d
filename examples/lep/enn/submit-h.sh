for REP in $(seq 1 3 ); do
        for CUTOFF in 06;  do
		for MAXNUM in 400; do
		SB_FILE=run_cutoff${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
		cp template-run.sb ${SB_FILE}
               	sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
                sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
		sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
		sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
		sbatch ${SB_FILE}
		done
        done
done

