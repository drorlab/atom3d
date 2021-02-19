# Submit training of Siamese networks on LBA data
for REP in $(seq 1 2); do
        for CUTOFF in 06; do
		for MAXNUM in 500 520 540 560 580 600; do
			SB_FILE=lba-id30-lmdb_cutoff-${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
			cp template-lba-id30-lmdb.sb ${SB_FILE}
        		sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
        		sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
			sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
			sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
			sbatch ${SB_FILE}
		done
	done
done

