for REP in $(seq 1 2); do
	SB_FILE=msp_rep${REP}.sb
	cp template-msp.sb ${SB_FILE}
	sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
	sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
	sbatch ${SB_FILE}
done

