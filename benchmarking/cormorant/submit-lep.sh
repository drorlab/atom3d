for REP in $(seq 1 2); do
        for CUTOFF in 06 07 08;  do
		for MAXNUM in 360 380 400 420 440 460 480; do
		SB_FILE=lep_cutoff${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
		cp template-lep.sb ${SB_FILE}
               	sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
                sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
		sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
		sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
		sbatch ${SB_FILE}
		done
        done
done

