for REP in $(seq 1 3 ); do
        for CUTOFF in 08;  do
		for MAXNUM in 400; do
		SB_FILE=run_cutoff${CUTOFF}_maxnum-${MAXNUM}_noh_rep${REP}.sb
		cp template-run.sb ${SB_FILE}
               	sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
                sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
		sed -i "s/ --load/-noh-rep${REP} --seed ${REP}${REP} --drop --load/g" ${SB_FILE}
		sed -i "s/-cormorant/-noh-cormorant-rep${REP}/g" ${SB_FILE}
		sbatch ${SB_FILE}
		done
        done
done

