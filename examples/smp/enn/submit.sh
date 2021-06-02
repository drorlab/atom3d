for LABEL in $( cat labels.txt); do
	echo "Submitting ${LABEL}"
	for REP in $(seq 1 5); do
		SB_FILE=run-${LABEL}_rep${REP}.sb
		cp template-run.sb ${SB_FILE}
		sed -i "s/TARGET/${LABEL}/g" ${SB_FILE}
		sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
		sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
		sbatch ${SB_FILE}
	done
done

