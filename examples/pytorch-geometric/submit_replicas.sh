for SB in *.sb; do
	for REP in $(seq 1 5); do
		cp $SB $( echo $SB | sed 's/.sb//g' )_rep${REP}.sb
	done
done

for REP in $(seq 1 5); do
	sed -i "s/ --load/-rep${REP} --seed ${REP}${REP} --load/g" *_rep${REP}.sb
        sed -i "s/ptg-qm9/ptg-qm9-rep${REP}/g" *_rep${REP}.sb
done
