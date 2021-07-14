for REP in $(seq 1 3 ); do
	for MAXNUM in 400; do
		for SAMPLES in 100; do
			SB_FILE=run_maxnum-${MAXNUM}_samples-${SAMPLES}_rep${REP}.sb
			cp template-run.sb ${SB_FILE}
                	sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
                	sed -i "s/SAMPLES/${SAMPLES}/g" ${SB_FILE}
			sed -i "s/ --load/_maxnum-${MAXNUM}_samples-${SAMPLES}_rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
			sbatch ${SB_FILE}
		done
	done
done

