import os
import datetime
import logging
import train_pdbbind
import argparse
import numpy as np

logger = logging.getLogger('pdbbind_log')


def main(base_dir='../../data/pdbbind/'):

	parser = argparse.ArgumentParser()
	parser.add_argument('split_method', type=str, default='random')
	args = parser.parse_args()

	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	log_dir = os.path.join(base_dir, 'logs', now)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	logging.basicConfig(filename=os.path.join(log_dir, f'train_{args.split_method}_cv_results.log'),level=logging.INFO)
	logger.info('{}\t{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format('fold', 'epoch', 'train_loss', 'val_loss', 'pearson', 'spearman'))

	val_losses = []
	rps = []
	rss = []
	for fold in range(10):
		print(f'training fold {fold}...')
		val_loss, r_p, r_s = train_pdbbind.main(fold=fold, split=args.split_method, log_dir=log_dir)
		val_losses.append(val_loss)
		rps.append(r_p)
		rss.append(r_s)
		print(f'\nFold {fold}: RMSE {val_loss}, Pearson R {r_p}, Spearman R {r_s}\n')

	# logger.info(f'\nAverage: RMSE {np.mean(val_losses)}, Pearson R {np.mean(rps)}, Spearman R {np.mean(rss)}')
	print(f'\nAverage: RMSE {np.mean(val_losses)}, Pearson R {np.mean(rps)}, Spearman R {np.mean(rss)}')


if __name__=="__main__":
    main()