# Submit training of ENN models on LBA data

CUTOFF=06
MAXNUM=600

for REP in $(seq 1 3); do

	SB_FILE=run-id30_cutoff-${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
	cp template-run-id30.sb ${SB_FILE}
       	sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
       	sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
	sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
	sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
	sbatch ${SB_FILE}

        SB_FILE=run-id60_cutoff-${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
        cp template-run-id60.sb ${SB_FILE}
        sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
        sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
        sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
        sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
        sbatch ${SB_FILE}

        SB_FILE=run-id30-siamese_cutoff-${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
        cp template-run-id30-siamese.sb ${SB_FILE}
        sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
        sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
        sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
        sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
        sbatch ${SB_FILE}

        SB_FILE=run-id60-siamese_cutoff-${CUTOFF}_maxnum-${MAXNUM}_rep${REP}.sb
        cp template-run-id60-siamese.sb ${SB_FILE}
        sed -i "s/MAXNUM/${MAXNUM}/g" ${SB_FILE}
        sed -i "s/CUTOFF/${CUTOFF}/g" ${SB_FILE}
        sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
        sed -i "s/-cormorant/-cormorant-rep${REP}/g" ${SB_FILE}
        sbatch ${SB_FILE}

done

