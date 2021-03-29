from sklearn.model_selection import GridSearchCV
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft 
import scipy.signal
from scipy import stats
from pprint import pprint
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
# for folder in os.listdir():	
# 	for file in os.listdir(folder):
# 		if file.endswith('.dat'):
# 			fname = f"{folder}\\{file}"
# 			x = np.fromfile(fname, dtype=dt)
# 			# with open(filename) as f:
# 			# 	lines = f.readlines()
# 			# 	# text = "".join(lines)
# 			# 	print(lines)
# 			# 	# print(text)
# 			# 	print('-----------------------------------------------------')

class SintecProj(object):
	"""docstring for SintecProj"""
	def __init__(self):
		self.fs = 125
		self.patient_path = str(os.getcwd())+'\\Patients'
		self.dataset_path = str(os.getcwd())+'\\Dataset'
		self.plot_setup()
		self.signal_list = ['3400715','3402408','3402291','3600293','3601140','3403232','3602237','3602666',#ok
							'3600376','3600490','3600620','3601272','3403274','3604404','3604430','3604660',#ok
							'3604404','3604430','3604660','3605744','3606315','3603256','3604217','3607634',#ok
							'3608436','3608706','3609155','3609182','3609463','3606882','3606909','3607464',#ok
							'3609839','3609868','3607711',#ok
							'3602521','3602766','3602772','3603658','3604352', #maybe
							'3605724','3606319','3606358','3606901','3607077'] #maybe

	def create_path(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def plot_setup(self):
		self.figsize = (15,9)
		self.create_path("Plots")
		self.plot_path = os.getcwd()+'\\Plots'
		plt.style.use('seaborn-darkgrid')

	def plot(self, df, pat_name):
		fig, axs = plt.subplots(2,1,sharex=True)
		fig.set_size_inches(self.figsize)
		axs[0].set_ylabel('ABP [mmHg]')
		axs[1].set_ylabel('[mV]')
		df['ABP'].plot(ax=axs[0])
		plt.suptitle(f'Patient: {pat_name}')
		df[['II','PLETH']].plot(ax=axs[1])
		plt.tight_layout()
		plt.savefig(f'{self.plot_path}\\{pat_name}.png')
		plt.close()
		if pat_name in self.signal_list: self.save_df(df,pat_name)

	def save_df(self,df,pat_name):
		self.create_path("Dataset")
		df.to_csv(f'{os.getcwd()}\\Dataset\\{pat_name}.csv')

	def data_reader(self):
		for file in os.listdir(self.patient_path):
			pat_name = file.split('_')[0]
			print(f'Patient: {pat_name}')
			df = pd.read_csv(f'{self.patient_path}\\{file}',quotechar="'",sep=',',skiprows=[1])
			print(df.iloc[0][0][0])

			if df.iloc[0][0][0] == '"':
				df.columns = [x.replace('"',"") for x in df.columns]
				df.columns = [x.replace("'","") for x in df.columns]

				df['Time'] = df['Time'].apply(lambda x: x[3:-2])
				df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: x[:-1])
				df = df.replace('-', np.nan)
				df.index = df['Time']

				def_columns = []
				for x in df.columns:
					if 'ABP' in x or x=='II' or 'PLETH' in x:
						def_columns.append(x)
				df = df[def_columns]
				df = df.astype(float)
				print(df)
				self.plot(df,pat_name)
			
			else:
				df['Time'] = df['Time'].apply(lambda x: x[1:-1])
				df.index = df['Time']
				df = df.replace('-', np.nan)
				df = df[['ABP','PLETH','II']]
				df = df.astype(float)
				print(df)
				self.plot(df,pat_name)			

	def peak_finder(self):
		self.create_path('Plots\\Peaks')
		tmp_path = self.plot_path+'\\Peaks'
		for file in os.listdir(self.dataset_path):
			patient = file.split('.')[0]
			print()
			print(f'Patient: {patient}')
			df = pd.read_csv(f'{self.dataset_path}\\{file}').dropna()
			df.index = range(0,len(df))
			b, a = scipy.signal.butter(N=5, 
				Wn=[1,10], 
				btype='band', 
				analog=False, 
				output='ba', 
				fs=125
				)	
			ecg_filt = scipy.signal.filtfilt(b,a,df['II'])
			ecg_diff = np.gradient(np.gradient(ecg_filt))

			ppg_filt = scipy.signal.filtfilt(b,a,df['PLETH'])

			#find DBP/SBP points
			DBPs,_ = scipy.signal.find_peaks(-df['ABP'],prominence=.5,distance=60,width=10)
			SBPs,_ = scipy.signal.find_peaks(df['ABP'],prominence=.5,distance=60,width=10)
			
			#find R peaks
			Rs,_ = scipy.signal.find_peaks(ecg_filt,distance=60) 
			Rs_diff,_ = scipy.signal.find_peaks(-ecg_diff,distance=60) #discarded bc of 3600490.csv

			#find SP peaks
			SPs,_ = scipy.signal.find_peaks(ppg_filt,prominence=.05,width=10)
			SPs_new = self.PPG_peaks_cleaner(ppg_filt, SPs)
			if False == True:
				# fig, axs = plt.subplots(3,2,sharex=True)
				fig.set_size_inches(self.figsize)
				gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1]) 
				axs[0,0] = plt.subplot(gs[0,0])
				axs[0,0].plot(df['ABP'],label='ABP')
				axs[0,0].scatter(DBPs,df['ABP'][DBPs],label='DBP',c='r')
				axs[0,0].scatter(SBPs,df['ABP'][SBPs],label='SBP',c='g')
				axs[0,0].set_ylabel('ABP[mmHg]')


				axs[1,0] = plt.subplot(gs[1])
				axs[1,0].plot(ecg_filt,label='ECG Filtered')
				# axs[1].plot(df['II'],label='ECG')
				axs[1,0].scatter(Rs_diff,ecg_filt[Rs_diff],label='R peaks - with gradient',s=100,c='r')
				axs[1,0].scatter(Rs,ecg_filt[Rs],label='R peaks',c='y')
				axs[1,0].set_ylabel('ECG [mV]')
				
				axs[2,0].plot(ppg_filt,label='PPG Filtered')
				# axs[2].plot(df['PLETH'],label='PPG')
				axs[2,0].scatter(SPs,ppg_filt[SPs],label='SP peaks',s=100,c='r')
				axs[2,0].scatter(SPs_new,ppg_filt[SPs_new],label='SP peaks - cleaned',c='y')
				axs[2,0].set_ylabel('PPG [mV]')

				[axs[x].legend(loc='lower left') for x in range(3)]
				x_ticks = np.arange(0,len(ppg_filt),125)
				axs[2,0].set_xticks(x_ticks)
				axs[2,0].set_xticklabels((x_ticks/self.fs).astype(int))
				axs[2,0].set_xlabel('Time [s]')
				print(f'ecg vector: {ecg_filt}')
				print(f'ppg vector: {ppg_filt}')
				plt.suptitle(f'Patient: {patient}')
				plt.tight_layout()
				plt.savefig(f'{tmp_path}\\{patient}')
			# plt.show()
			dataset = self.find_PTT(ecg_filt,Rs,ppg_filt,SPs_new,patient)
			df.index = np.array(list(df.index))/self.fs
			print(df)
			dataset['DBP'] = df['ABP'].iloc[DBPs]
			dataset['SBP'] = df['ABP'].iloc[SBPs]
			regr_path = self.dataset_path+'\\Regression'
			self.create_path(regr_path)
			dataset.to_csv(f'{regr_path}\\{patient}.csv')

	def PPG_peaks_cleaner(self, ppg, SP):
		x_ppg = np.arange(min(ppg),max(ppg),.001)
		kde_ppg = stats.gaussian_kde(ppg)
		kde_sp = stats.gaussian_kde(ppg[SP])

		peak_sp,_ = scipy.signal.find_peaks(kde_sp(x_ppg))
		n_peaks = len(peak_sp)
		if n_peaks == 2: 
			sp_idx = np.argmin(kde_sp(x_ppg)[peak_sp])
			# peak_sp = peak_sp[sp_idx]

			# plt.plot(kde_sp(x_ppg)[peak_sp[0]:peak_sp[1]])
			minimum = scipy.signal.find_peaks(-kde_sp(x_ppg)[peak_sp[0]:peak_sp[1]])
			minimum = -x_ppg[minimum[0]]
			# exit()

			SP = [x for x in SP if ppg[x] > int(minimum[0])]

		#TODO: plot in same graph - rotated graph
		# plt.figure()
		# plt.plot(x_ppg, kde_ppg(x_ppg),label='ppg distribution')
		# plt.plot(x_ppg, kde_sp(x_ppg),label='SP-peaks distribution')
		# plt.scatter(peak_sp,c='r',label='SP peaks center')
		# plt.legend()
		return SP

	def find_PTT(self,ECG,ECG_peaks,PPG,PPG_peaks,patient):
		self.create_path('Plots\\HR and PTT')
		tmp_path = self.plot_path+'\\HR and PTT'
		#ECG_peaks,PPG_peaks: vectors containig indices of peaks 
		#ECG,PPG: vectors containig ECG/PPG curves 

		#transofrm in time series:
		ecg_TS = np.array(ECG_peaks)/self.fs
		ppg_TS = np.array(PPG_peaks)/self.fs
		print(f'ECG in seconds:{ecg_TS}')
		print(f'PPG in seconds:{ppg_TS}')

		#HR evaluation:
		HR = 60/(ecg_TS[1::] -ecg_TS[0:-1])
		print(f'HR:{HR}')

		#PTT evaluation:
		time = np.arange(0,max(max(ECG_peaks),max(PPG_peaks)),1)
		real_time = time/self.fs

		y = time*0
		for k in time:
			if time[k] in ECG_peaks:
				y[k]=1
			if time[k] in PPG_peaks:
				y[k]=2

		index = np.argwhere(y>0).flatten()
		yy = list(y[index])
		time1 = time[index]

		results = []
		b = [1,2]
		results = [i for i in range(len(yy)) if yy[i:i+len(b)] == b] 
		index_rf = time1[np.array(results)]
		index_spf = time1[np.array(results)+1]

		time_rf = real_time[index_rf]
		time_spf = real_time[index_spf]

		rf_diff = np.diff(time_rf)
		ptt = time_spf - time_rf
		# print(f'PTT: {ptt}')
		plt.close('all')
		fig, axs = plt.subplots(2,1,sharex=True)
		fig.set_size_inches(self.figsize)

		axs[0].plot(ecg_TS[1:],HR,label='Heart Rate')

		#HR cleaning:
		if np.std(HR) > 3:
			up_bound, low_bound = np.mean(HR)+np.std(HR), np.mean(HR)-np.std(HR)
			axs[0].axhline(np.mean(HR),c='r',lw=4, label='Mean Value')
			axs[0].fill_between(ecg_TS[1:], low_bound, up_bound, alpha=0.15, color='red', lw=4)
			nan_idx = np.concatenate((np.argwhere(HR<=low_bound),np.argwhere(HR>=up_bound)))
			HR[nan_idx] = np.nan		
			HR = pd.DataFrame(HR).interpolate(method='polynomial',order=2)
			axs[0].plot(ecg_TS[1:],HR.values.tolist(),label='HR - cleaned',c='g')
		axs[0].legend()

		axs[1].plot(ecg_TS,ECG[ECG_peaks],'o-',label='R peaks')
		axs[1].plot(real_time[index_rf],ECG[index_rf],'o-',label='R peaks - newly found')
		axs[1].plot(ppg_TS,PPG[PPG_peaks],'o-',label='SP peaks')
		axs[1].plot(real_time[index_spf],PPG[index_spf],'o-',label='SP peaks - newly found')
		axs[1].hlines(ptt, xmin=real_time[index_rf], xmax=real_time[index_spf], colors='red', linestyles='solid', label='ptt')
		axs[1].legend()
		plt.tight_layout()
		plt.savefig(f'{tmp_path}\\{patient}')

		tmp_df_hr = pd.DataFrame(HR)
		tmp_df_hr.index = ecg_TS[1:]
		print(tmp_df_hr)

		tmp_df_ptt = pd.DataFrame(ptt)
		tmp_df_ptt.index = real_time[index_rf]
		
		df = pd.DataFrame({'Time':real_time})
		df.index = df['Time']
		df['HR'] = tmp_df_hr
		df['PTT'] = tmp_df_ptt
		return df.drop('Time',axis=1)

	def regression_process(self):
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression

		TEST_PERC = .75
		regr_path = 'Dataset\\Regression'
		for file in os.listdir(regr_path):
			patient = file.split('.')[0]
			df = pd.read_csv(regr_path+'\\'+file).set_index('Time')
			df = df.dropna(how='all')
			ax = df.plot(style='o')
			df = df.interpolate(method='spline',order=2).dropna()
			df.plot(ax=ax)
			plt.close()
			print(df)

			test_size = int(TEST_PERC*len(df.index))
			fig, axs = plt.subplots(3,1)
			fig.set_size_inches((16,9))

			#DBP Prediction
			X_train_dbp,y_train_dbp = df[['HR','PTT']].iloc[0:test_size], df['DBP'].iloc[0:test_size]
			print(X_train_dbp)
			X_test_dbp,y_test_dbp = df[['HR','PTT']].iloc[test_size::], df['DBP'].iloc[test_size::]
			#SBP Prediction
			X_train_sbp,y_train_sbp = df[['HR','PTT']].iloc[0:test_size], df['SBP'].iloc[0:test_size]
			X_test_sbp,y_test_sbp = df[['HR','PTT']].iloc[test_size::], df['SBP'].iloc[test_size::]
			
			axs[0].plot(y_train_dbp,label='train')
			axs[1].plot(y_train_sbp,label='train')
			maes_dbp, maes_sbp = [],[]

			#Polynomial regression
			# pol_orders = [1,2,3,4]
			# for order in pol_orders:
			# 	polynomial_features= PolynomialFeatures(degree=order)

			# 	x_poly_dbp = polynomial_features.fit_transform(X_train_dbp)
			# 	model_dbp = LinearRegression()
			# 	model_dbp.fit(x_poly_dbp, y_train_dbp)
			# 	y_poly_pred = model_dbp.predict(polynomial_features.fit_transform(X_test_dbp))
			# 	MAE_dbp = round(mean_absolute_error(y_test_dbp, y_poly_pred),2)
			# 	maes_dbp.append(MAE_dbp)
			# 	axs[0].plot(y_test_dbp.index,y_poly_pred,label=f'Polynomial, deg:{order}')


			# 	x_poly_sbp = polynomial_features.fit_transform(X_train_sbp)
			# 	model_sbp = LinearRegression()
			# 	model_sbp.fit(x_poly_sbp, y_train_sbp)
			# 	y_poly_pred = model_sbp.predict(polynomial_features.fit_transform(X_test_sbp))
			# 	MAE_sbp = round(mean_absolute_error(y_test_sbp, y_poly_pred),2)
			# 	maes_sbp.append(MAE_sbp)
			# 	axs[1].plot(y_test_sbp.index,y_poly_pred,label=f'Polynomial, deg:{order}')

			#Support Vector Regression
			pol_orders = []
			Cs = [1,50,1000]
			for c in Cs:
				regr = SVR(C=c, epsilon=0.2)
				regr.fit(X_train_dbp, y_train_dbp)
				y_hat_dbp = regr.predict(X_test_dbp)
				axs[0].plot(y_test_dbp.index,y_hat_dbp,label=f'SVR, deg:{c}')
				MAE_dbp = round(mean_absolute_error(y_test_dbp, y_hat_dbp),2)
				maes_dbp.append(MAE_dbp)

				regr = SVR(C=c, epsilon=1)
				regr.fit(X_train_sbp, y_train_sbp)
				y_hat_sbp = regr.predict(X_test_sbp)
				axs[1].plot(y_test_sbp.index,y_hat_sbp,label=f'SVR, deg:{c}')
				MAE_sbp = round(mean_absolute_error(y_test_sbp, y_hat_sbp),2)
				maes_sbp.append(MAE_sbp)

			#grid search SVR
			parameters = {'kernel':('linear', 'rbf'), 'epsilon':np.linspace(.1,5,5), 'C':np.linspace(.1,1000,10)}
			svr = SVR()
			clf = GridSearchCV(svr, parameters)
			clf.fit(X_train_dbp, y_train_dbp)
			y_hat_dbp = clf.predict(X_test_dbp)
			axs[0].plot(y_test_dbp.index,y_hat_dbp,label=f'SVR: Best')
			print(f'Best for DBP: {clf.best_params_}')
			MAE_dbp = round(mean_absolute_error(y_test_dbp, y_hat_dbp),2)
			maes_dbp.append(MAE_dbp)

			clf = GridSearchCV(svr, parameters)
			clf.fit(X_train_sbp, y_train_sbp)
			y_hat_sbp = clf.predict(X_test_sbp)
			print(f'Best for SBP: {clf.best_params_}')
			axs[1].plot(y_test_sbp.index,y_hat_sbp,label=f'SVR: Best')
			MAE_sbp = round(mean_absolute_error(y_test_sbp, y_hat_sbp),2)
			maes_sbp.append(MAE_sbp)

			#Ridge Regression
			alphas = [.01, 10, 100, 10000]
			for alpha in alphas:
				y_hat_dbp = self.regression(y_train_dbp,X_train_dbp,X_test_dbp, alpha)
				axs[0].plot(y_test_dbp.index,y_hat_dbp,label=f'alpha = {alpha}')
				MAE_dbp = round(mean_absolute_error(y_test_dbp, y_hat_dbp),2)
				maes_dbp.append(MAE_dbp)

				y_hat_sbp = self.regression(y_train_sbp,X_train_sbp,X_test_sbp, alpha)
				axs[1].plot(y_test_sbp.index,y_hat_sbp,label=f'alpha = {alpha}')
				MAE_sbp = round(mean_absolute_error(y_test_sbp, y_hat_sbp),2)
				maes_sbp.append(MAE_sbp)

			
			width = 0.35 
			axis = np.arange(len(maes_dbp))
			axs[2].bar(axis+width/2,maes_dbp,width,label='DBP')
			axs[2].bar(axis-width/2,maes_sbp,width,label='SBP')
			axs[2].set_ylim(0,10)
			axs[2].set_title('MAEs')
			x_labs = [f'POL: {x}' for x in pol_orders]
			[x_labs.append(f'SVR: {x}') for x in Cs]
			x_labs.append('SVR: Best')
			[x_labs.append(f'RIDGE: {x}') for x in alphas]
			axs[2].set_xticks(range(len(x_labs)))
			axs[2].set_xticklabels((x_labs))
			axs[2].set_ylabel('MAE [-]')
			axs[2].legend()

			axs[0].plot(y_test_dbp,label='test')
			axs[0].set_ylabel('DBP [mmHg]')
			axs[0].set_xlabel('Time [s]')
			axs[0].set_title('Prediction vs. Test')
			axs[0].set_ylim(min(df['DBP'])-10,max(df['DBP'])+10)
			axs[0].legend(ncol=3)

			axs[1].plot(y_test_sbp,label='test')
			axs[1].set_ylabel('SBP [mmHg]')
			axs[1].set_xlabel('Time [s]')
			axs[1].set_ylim(min(df['SBP'])-10,max(df['SBP'])+10)
			axs[1].legend(ncol=3)
			# plt.title(f'Alpha = {alpha}')
			# ax = df.plot(style='o-')
			# df_interpolated.plot(ax=ax)
			# plt.legend()
			plt.tight_layout()
			# plt.show()
			self.create_path('Plots\\Regression')
			plt.savefig(f'Plots\\Regression\\{patient}.png')

	def regression(self,y_train,X_train,X_test, alpha):
		clf = Ridge(alpha=alpha)
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)
		return pred

SP = SintecProj()
# SP.data_reader()
# SP.peak_finder()
SP.regression_process()