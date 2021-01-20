for REP in $(seq 1 5); do
	SB_FILE=lba_rep${REP}.sb
	cp template-lba.sb ${SB_FILE}
	sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
	sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
	sbatch ${SB_FILE}
done

