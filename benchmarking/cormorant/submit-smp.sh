#for LABEL in $( cat labels-smp.txt); do 
for LABEL in alpha; do
	echo "Submitting ${LABEL}"
	for REP in $(seq 1 5); do
		SB_FILE=smp-${LABEL}_rep${REP}.sb
		cp smp_template.sb ${SB_FILE}
		sed -i "s/TARGET/${LABEL}/g" ${SB_FILE}
		sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" ${SB_FILE}
		sed -i "s/cormo-smp/cormo-smp-rep${REP}/g" ${SB_FILE}
		sbatch ${SB_FILE}
	done
done

