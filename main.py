#%% User: danru
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import GridSearchCV, RepeatedKFold
warnings.simplefilter("ignore")
import scipy.fft 
import scipy.signal
# import MIMICUtil as ut
# import heartpy as hp
# from sklearn.preprocessing import StandardScaler
# from filterpy.kalman import KalmanFilter

class DataRegressor(object):
	"""docstring for DataReader"""
	def __init__(self):
		self.setup()
		patient_n = 4
		plt.style.use('seaborn-darkgrid')
		# self.patient = self.patients[patient_n].split("'")[1]
		self.STEP = 0.1
		self.RS = 7
		self.dict_ = {'PLETH':'PPG','II':'ECG'}
		self.plot_peaks = False
		self.plot = False	
		self.savefile = True
		np.random.seed(self.RS) 

	def create_path(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def setup(self):
		# with open("Folder\\config.txt") as file:
		# 	config = file.read()
		# self.HeightSPmin = 0.5
		# self.HeightSPmax = 2
		self.fs0 = 125
		self.HeightRmin = .6
		self.HeightRmax = 4

		self.CCStdCoef = 0
		self.NPEAK = 200
		self.LENSEGM = 10 
		self.K = 5
		self.Z_SCORE_train = 5
		self.Z_SCORE_test = 5
		self.NSR_THRESHOLD = 30

		self.CORR_THRESHOLD = 50 
		self.SOL_WEIGHT = 5
		self.WindLenDefault = 650

		self.SBP_MAX = 190
		self.SBP_MIN = 80
		self.DBP_MAX = 79
		self.DBP_MIN = 45

		self.WidthPPGDefaultMin = 100
		self.WidthPPGDefaultMax = 650

		self.dTps = 1500
		self.dTes = 700
		self.dTpe = 800

		self.dTepmax = 1000
		self.dTepmin = 330
		self.dTsdmin = 100
		self.dTsdmax = 1300

		self.wind_len = self.WindLenDefault
		self.RHeightCoeff = 1

		self.WidthABPMin = 0.5
		self.WidthABPMax = 950

		with open("Folder\\patients.txt") as file:
			self.patients = file.readlines()

		self.create_path("Dataset")
		self.create_path("Plots")
		# exit()

	def interpolation(self):
		for e in ['PLETH','II','ABP']:
			df = pd.read_csv(f'Dataset\\{e}.csv').set_index('Unnamed: 0')
			df = df.interpolate()
			df.to_csv('Dataset\\Interpolate'+e+'.csv')

	def signaltonoise(self, a, axis=0, ddof=0):
		a = np.asanyarray(a)
		m = a.mean(axis)
		sd = a.std(axis=axis,ddof=ddof)
		return np.where(sd == 0, 0, m/sd)

	def cleanWindRSP(self, R, SP, ecg, ppg, time, dTepmin=100,dTepmax=1000):
		"""
		R/SP: indices of peaks 
		ecg/ppg: y values (i.e. ecg[x] == 0.2mV)
		time: x values (i.e. time[x] == 20s)
		"""
		# plt.fill_between(x_integral, 0, kde(x_integral),
		# alpha=0.3, color='b', label="Area: {:.3f}".format(integral))
		SP, R = list(SP), list(R)
		if SP[0] < R[0]:
			SP.pop(0)

		fig, axs = plt.subplots(2,1,sharex=True)

		x_ppg = np.arange(min(ppg),max(ppg),.001)
		x_ecg = np.arange(min(ecg),max(ecg),.001)
		SP = list([x for x in SP if round(ppg[x],3)!=0])
		R = list([x for x in R if round(ecg[x],3)!=0])

		kde_ppg = stats.gaussian_kde(ppg)
		kde_sp = stats.gaussian_kde(ppg[SP])
		kde_ecg = stats.gaussian_kde(ecg)
		kde_r = stats.gaussian_kde(ecg[R])

		peak_r, _ = scipy.signal.find_peaks(kde_r(x_ecg))
		r_idx = np.argmax(kde_r(x_ecg)[peak_r])
		peak_r = peak_r[r_idx]

		peak_sp, _ = scipy.signal.find_peaks(kde_sp(x_ppg))
		sp_idx = np.argmax(kde_sp(x_ppg)[peak_sp])
		peak_sp = peak_sp[sp_idx]

		x_r_min,x_r_max = x_ecg[peak_r]*.8, x_ecg[peak_r]*1.2
		x_sp_min,x_sp_max = x_ppg[peak_sp]*.7, x_ppg[peak_sp]*1.3

		# axs[0].plot(x_ecg,kde_ecg(x_ecg),label='ecg')
		# axs[0].plot(x_ecg,kde_r(x_ecg),label='R')
		# axs[0].scatter(x_ecg[peak_r],kde_r(x_ecg)[peak_r],c='red')
		# axs[0].axvline(x_r_min)
		# axs[0].axvline(x_r_max)
		# axs[0].legend()
		
		# axs[1].plot(x_ppg,kde_ppg(x_ppg),label='ppg')
		# axs[1].plot(x_ppg,kde_sp(x_ppg),label='SP')
		# axs[1].scatter(x_ppg[peak_sp],kde_sp(x_ppg)[peak_sp],c='red')
		# axs[1].axvline(x_sp_min)
		# axs[1].axvline(x_sp_max)
		# axs[1].legend()
		# plt.show()
		R_f = [x for x in R if x_r_min<=ecg[x]<=x_r_max]	
		SP_f = [x for x in SP if x_sp_min<=ppg[x]<=x_sp_max]	
		return R_f, SP_f

	def find_ptt(self, sbp, dbp, Rf, SPf, timestamp, abp, ecg, ppg, batch):
		""" 
			dTps : time delay between SBP point and SP point
			dTes : time delay between SBP point and R peak
			dTpe : time delay between R point and SP 
			sbp/dbp : list of SBP/DBP indexes
			Rf : list of R peak indexes
			SPf : list of SP indexes
			timestamp/abp/ecg/ppg : series
			patient : string """

		flag = True
		Mptt,Mhr,Msbp,Mdbp = [], [], [], []
		# sbp_val = abp[sbp]
		# dbp_val = abp[dbp]

		plt.close('all')
		fig, axs = plt.subplots(2,1, sharex=True) 
		fig.set_size_inches((16,9))
		axs[0].plot(dbp,abp[dbp],'r^',label = 'DBP')
		axs[0].plot(sbp,abp[sbp],'y^',label = 'SBP')
		axs[1].plot(SPf,ppg[SPf],'mx',label = 'SP')
		axs[1].plot(Rf,ecg[Rf],'gx',label = 'R')

		time = np.arange(0,7500,1)
		real_time = time/self.fs0
		y = time*0
		cnt = 0
		for k in range(len(time)):
			# if time[k] in sbp:
			# 	y[k]=1
			if time[k] in Rf:
				y[k]=1
			if time[k] in SPf:
				y[k]=2

		index = np.argwhere(y>0).flatten()
		yy = list(y[index])
		time1 = time[index]

		results = []
		b = [1,2]
		# b = [1,2,3]
		results = [i for i in range(len(yy)) if yy[i:i+len(b)] == b]
		if results != []:
			#indices of yy where the terna start
			# index_sbp = time1[results]
			index_rf = time1[np.array(results)]
			index_spf = time1[np.array(results)+1]

			# time_sbp = real_time[index_sbp]
			time_rf = real_time[index_rf]
			time_spf = real_time[index_spf]

			rf_diff = np.diff(time_rf)
			ptt = time_spf - time_rf

			try:
				axis = np.arange(0,5,.01)
				kde = stats.gaussian_kde(rf_diff)
				peaks,_ = scipy.signal.find_peaks(kde(axis))
			except:
				peaks = [] 
			
			hr,index_hr=[],[]

			if len(peaks)>1:
				for x,y in zip(rf_diff, index_rf):
					if x < np.mean(rf_diff)*1.4:
						hr.append(60/x)
						index_hr.append(y)
				# hr = np.array([60/x for x in rf_diff if x < np.mean(rf_diff)*1.4])
				# index_hr = np.array([x for x in rf_diff if x < np.mean(rf_diff)*1.4])
			else:
				hr = 60/rf_diff
				index_hr = index_rf[:-1]

			hr,index_hr = np.array(hr), np.array(index_hr)

			# DBP,x_DBP = [],[]
			# for i in time_sbp:
			# 	mask = [timestamp[dbp]>i]
			# 	if np.array(dbp)[mask] != []:
			# 		tmp = real_time[np.array(dbp)[mask][0]]
			# 		idx = np.array(dbp)[mask][0]
			# 		dbp_val = abp[np.array(dbp)[mask][0]]
			# 		if (tmp - i) < np.mean(hr/(2*60))*1.1:
			# 			DBP.append(dbp_val)
			# 			x_DBP.append(idx)

			axs[1].hlines(ptt, xmin=index_rf, xmax=index_spf, colors='red', linestyles='solid', label='ptt')
			axs[0].legend(loc='upper right')
			axs[1].legend(loc='upper right')
			title = 'Corresponding values'
			fig.suptitle(title)
			plt.xlabel('Indices')
			axs[0].set_ylabel('ABP [mmHg]')
			axs[1].set_ylabel('PTT [ms]')
			plt.tight_layout()
			plt.savefig(f'Plots\\Find PTT\\{batch}_{title}.png') 
			
			# SBP = abp[index_sbp]
			# x_SBP = index_sbp
			# SBP[:] = np.nan
			# sbp_tmp = abp[index_sbp]
			
			HR = hr
			x_HR = index_hr
			# HR[:] = np.nan

			PTT = ptt
			x_PTT = index_rf
 			# PTT[:] = np.nan

			# try:
			# 	PTT[index_rf] = ptt
			# 	SBP[index_sbp] = sbp_tmp
			# 	HR[index_hr] = hr
			# except:
			# 	pass

		else: 
			flag = False
			PTT,HR = [], []
			x_PTT,x_HR = [], []

		if len(PTT)<5 or len(HR)<5:
			flag = False
		
		print(len(PTT),len(x_PTT))

		return PTT,x_PTT, HR,x_HR,flag

	def cleanWindDSBP(self,dbp,sbp,dTsdmin=100,dTsdmax=1500):
		""" Clean windows : R and SP - Retain only Cardiac cycles where both R and SP valid points are found.
		Search startts from the first R peak; the following SP point has to fall in an interval after R defined by dTepmin and 
		dTepmax. Which means that the SP that corresponds to a certain R peak must have a certain Time Delay with respect to it."""
		"""
		SP : list of indexes of SP points
		R : list of indexes of R peaks
		ppg/ecg : series
		window_len : float, length of the cardiac cycle
		patient : string, patient's code in MIMIC
		dTepmin : minimum time delay between R and corresponding SP
		dTepmax : maximum time delay between R and corresponding SP """
		sbp = list(sbp)
		dbp = list(dbp)
		dbp_f = []
		sbp_f = []

		""" start from the first SBP point """
		if dbp[0] < sbp[0]:
			dbp.pop(0)
		start = 0 
		for i in range(len(sbp)-1):
			found = 0
			j = start
			while found == 0 and j< len(dbp):
				if dbp[j] > sbp[i] +dTsdmin and dbp[j] < sbp[i] + dTsdmax:
					found = 1
					start = j+1
					dbp_f.append(dbp[j])
					sbp_f.append(sbp[i])
				else:
					found = 0
					j+=1
		return dbp_f,sbp_f

	def SBP_DBP_values(self):
		ppg,ecg,abp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
		pat_dict = {}
		"""
		pat_dict = {
			'3002511':{
				'1':{
					'PTT':[],
					'HR':[],
					'DBP':[],
					'SBP':[]
					}
				}
		}
		"""
		for x,y in zip([ppg,ecg,abp],['PLETH','II','ABP']):
			df = pd.read_csv(f'Dataset\\Interpolate{y}.csv').set_index('Unnamed: 0')
			if y.startswith('P'):
				ppg = df
			elif y.startswith('I'):
				ecg = df
			else:
				abp = df
		# index = list(ppg.index)
		# Find length of cardiac cycle 
		# wind_len = ut.findCC(ecg_int,self.WindLenDefault,self.WindStdCoef,patient)
		
		iter_lst = ppg.columns
		# iter_lst = ppg.columns[:4]

		pat_tmp = self.patients[19].split("'")[-2]
		print(pat_tmp)
		for col in iter_lst:
			print(f'******************** Batch: {col} ********************')
			# filt = True

			filt = pat_tmp in col
			# filt = '3001023' in col
			# filt = '3000393' not in col and 
			patient, batch = col.split('_')
			try:
				temporary = pat_dict[patient]
			except:
				pat_dict.update({patient:{}})

			if filt:
				print('Number of NaNs:')
				print(f'ECG: {ecg[col].isna().sum()}')
				print(f'PPG: {ppg[col].isna().sum()}')
				print(f'ABP: {abp[col].isna().sum()}')
				ABP = abp[col].dropna()
				index = list(ppg[col].dropna().index)
				index = np.array([round(x,3) for x in index])
				ECG, PPG = np.array(list(ecg[col].dropna())),np.array(list(ppg[col].dropna())) 
				ABP = np.array(ABP)

				if sum(ABP) != 0:
					b, a = scipy.signal.butter(N = 4, Wn=[.5,10], btype='bandpass', analog=False, output='ba', fs=125)	

					ecg_filt = scipy.signal.filtfilt(b,a,ECG)
					ppg_filt = scipy.signal.filtfilt(b,a,PPG)
					ecg_diff = np.gradient(np.gradient(ecg_filt))
					
					Rs,_ = scipy.signal.find_peaks(-ecg_diff,distance=60)
					SPs,_ = scipy.signal.find_peaks(ppg_filt,prominence=.05,width=10)
					DBPs,_ = scipy.signal.find_peaks(-ABP,prominence=.5,distance=60,width=10)
					SBPs,_ = scipy.signal.find_peaks(ABP,prominence=.5,distance=60,width=10)

					if len(DBPs)!=0 and len(SBPs)!=0 and len(Rs)!=0 and len(SPs)!=0:
						DBPs_f, SBPs_f = self.cleanWindDSBP(DBPs, SBPs)
						Rs_f, SPs_f = self.cleanWindRSP(Rs, SPs,ecg_filt,ppg_filt,index)
						pat_dict[patient].update({batch:{}})
						# print('Rs:',np.array(Rs))
						# print('SPs:',np.array(SPs))
						# print('DBPs:',np.array(DBPs))
						# print('SBPs:',np.array(SBPs))

						x_R = [index[x] for x in Rs]
						x_R_f = [index[x] for x in Rs_f]

						x_SP = [index[x] for x in SPs]
						x_SP_f = [index[x] for x in SPs_f]

						x_SBP = [index[x] for x in SBPs]
						x_SBP_f = [index[x] for x in SBPs_f]

						x_DBP = [index[x] for x in DBPs]
						x_DBP_f = [index[x] for x in DBPs_f]

						fig, axs = plt.subplots(3,1,sharex=True) 
						fig.set_size_inches((16,9))
						fig.suptitle(col)
						axs[0].plot(index,ecg_filt)
						axs[0].set_ylabel('ECG [mV]')
						axs[0].scatter(x_R, ecg_filt[Rs],color='red',label='R peak') 
						axs[0].scatter(x_R_f, ecg_filt[Rs_f],color='yellow',label='R peak') 
						# axs[0].legend('lower right')

						axs[1].plot(index,ppg_filt)
						axs[1].set_ylabel('PPG [mV]')
						axs[1].scatter(x_SP,ppg_filt[SPs],color='red',label='S peak') 
						axs[1].scatter(x_SP_f,ppg_filt[SPs_f],color='yellow',label='S peak') 

						axs[2].set_ylabel('ABP [mmHg]')
						axs[2].set_xlabel('Time [s]')
						try:
							axs[2].plot(index,ABP)
							axs[2].scatter(x_DBP,ABP[DBPs],color='red',label='DBP')
							axs[2].scatter(x_DBP_f,ABP[DBPs_f],color='yellow',label='DBP_f')

							axs[2].scatter(x_SBP,ABP[SBPs],color='green',label='SBP')
							axs[2].scatter(x_SBP_f,ABP[SBPs_f],color='blue',label='SBP_f')
						except ValueError:
							pass

						plt.tight_layout()
						# plt.show()
						if self.savefile:
							plt.savefig(f'Plots\\Batches\\{col}.png')

						Mptt,x_PTT,Mhr,x_HR,flag = self.find_ptt(SBPs_f, DBPs_f, Rs_f, SPs_f, index, ABP, ecg_filt, ppg_filt, col)
						Mptt,Mhr,Msbp,Mdbp,x_HR,x_SBP,x_DBP,x_PTT = list(Mptt),list(Mhr),list(ABP[SBPs_f]),list(ABP[DBPs_f]), list(x_HR),list(SBPs_f),list(DBPs_f),list(x_PTT)

						if flag:
							pat_dict[patient][batch].update({'PTT':Mptt})
							pat_dict[patient][batch].update({'HR':Mhr})
							pat_dict[patient][batch].update({'SBP':Msbp})
							pat_dict[patient][batch].update({'DBP':Mdbp})
							
							pat_dict[patient][batch].update({'x_PTT':x_PTT})
							pat_dict[patient][batch].update({'x_HR':x_HR})
							pat_dict[patient][batch].update({'x_SBP':x_SBP})
							pat_dict[patient][batch].update({'x_DBP':x_DBP})

						else:
							pat_dict[patient][batch].update({'PTT':[None]})
							pat_dict[patient][batch].update({'HR':[None]})
							pat_dict[patient][batch].update({'SBP':[None]})
							pat_dict[patient][batch].update({'DBP':[None]})

							pat_dict[patient][batch].update({'x_PTT':[None]})
							pat_dict[patient][batch].update({'x_HR':[None]})
							pat_dict[patient][batch].update({'x_SBP':[None]})
							pat_dict[patient][batch].update({'x_DBP':[None]})
		# print(pat_dict)
		PAT_ITER = []
		for patient in pat_dict.keys():
			PTT,HR,SBP,DBP = [], [], [], []
			x_PTT,x_HR,x_SBP,x_DBP = [], [], [], []
			if bool(pat_dict[patient].keys()):
				PAT_ITER.append(patient)

		for patient in PAT_ITER:
			batches = pat_dict[patient].keys()
			[PTT.extend(pat_dict[patient][x]['PTT']) for x in batches]
			[HR.extend(pat_dict[patient][x]['HR']) for x in batches]
			[SBP.extend(pat_dict[patient][x]['SBP']) for x in batches]
			[DBP.extend(pat_dict[patient][x]['DBP']) for x in batches]

			[x_PTT.extend(pat_dict[patient][x]['x_PTT']) for x in batches]
			[x_HR.extend(pat_dict[patient][x]['x_HR']) for x in batches]
			[x_SBP.extend(pat_dict[patient][x]['x_SBP']) for x in batches]
			[x_DBP.extend(pat_dict[patient][x]['x_DBP']) for x in batches]

			print('PTT:',len(PTT),len(x_PTT))
			print('HR:',len(HR),len(x_HR))
			print('SBP:',len(SBP),len(x_SBP))
			print('DBP:',len(DBP),len(x_DBP))

			PTT_df = pd.DataFrame({'PTT':PTT,'x_PTT':x_PTT})
			HR_df = pd.DataFrame({'HR':HR,'x_HR':x_HR})
			SBP_df = pd.DataFrame({'SBP':SBP,'x_SBP':x_SBP})
			DBP_df = pd.DataFrame({'DBP':DBP,'x_DBP':x_DBP})

			self.create_path("Dataset\\Patients")
			PTT_df.to_csv('Dataset\\Patients\\PTT_'+patient+'.csv')
			HR_df.to_csv('Dataset\\Patients\\HR_'+patient+'.csv')
			SBP_df.to_csv('Dataset\\Patients\\SBP_'+patient+'.csv')
			DBP_df.to_csv('Dataset\\Patients\\DBP_'+patient+'.csv')

				# print(ecg_filt)
				# print(ppg_filt)
				# win_len = 125
				# Rs,SPs,DBPs,SBPs = [], [], [], []
				# NSR_a,NSR_e,NSR_p,TIME_NSR = [], [], [],[]
				# st_win, end_win = 0,win_len
				# condition = True
				# while condition and ecg[col].isna().sum()<300:
				# 	ecg_filt_win = ecg_filt[st_win:end_win]
				# 	ppg_filt_win = ppg_filt[st_win:end_win]
				# 	abp_win = ABP[st_win:end_win]
				# 	ecg_diff = np.gradient(np.gradient(ecg_filt_win))
				# 	# print(ecg_filt_win)
				# 	# print(ecg_diff)
				# 	nsr_ecg = self.signaltonoise(ecg_filt_win)
				# 	nsr_ppg = self.signaltonoise(ppg_filt_win)
				# 	nsr_abp = self.signaltonoise(abp_win)
				# 	NSR_a.append(nsr_abp)
				# 	NSR_p.append(nsr_ppg)
				# 	NSR_e.append(nsr_ecg)
				# 	TIME_NSR.append(index[st_win])
					# if nsr_ppg <= 30 and nsr_ecg <= 30:
						# for x,y in zip([Rs, SPs, DBPs, SBPs],[R,SP,DBP,SBP]):
						# 	x.append(y)
						# Rs.extend(R+st_win)
						# SPs.extend(SP+st_win)
						# DBPs.extend(DBP+st_win)
						# SBPs.extend(SBP+st_win)

					# else:
					# 	print('-------------------------------------------')
					# 	print('NSR ECG:',nsr_ecg)
					# 	print('NSR PPG:',nsr_ppg)
					# 	print('NSR PPG:',nsr_abp)
					# 	print('-------------------------------------------')
					# st_win = end_win
					# end_win+=win_len
					# if end_win == len(ecg_filt):
					# 	condition = False

	def BP_retrival(self):
		for patient in os.listdir('Dataset\\Patients'):
			df = pd.read_csv(f'Dataset\\Patients\\{patient}')
			print(patient)
			split = int(.75*len(df))
			X_train, y_train = df.iloc[:split].drop(['SBP','DBP'],axis=1), df.iloc[:split][['SBP','DBP']]
			X_test, y_test = df.iloc[split-1:].drop(['SBP','DBP'],axis=1), df.iloc[split-1:][['SBP','DBP']]
			# X, y = df.drop(['SBP','DBP'],axis=1), df[['SBP','DBP']]
			# for solver in ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']:
			# fig, axs = plt.subplots(4,1,sharex=True) 
			# fig.set_size_inches((16,9))

			cnt=0
			model = Ridge()
			cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
			grid = dict()
			grid['alpha'] = np.arange(0, 1, 0.01)
			search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
			results = search.fit(X_train, y_train)

			y_hat = search.predict(X_test)
			alpha = results.best_params_['alpha']

			print('MAE: %.3f' % results.best_score_)
			print('Config: %s' % results.best_params_)

			# clf = Ridge(alpha=alpha)
			# clf.fit(X_train, y_train)
			# y_hat = clf.predict(X_test)

			y_hat = pd.DataFrame(y_hat)
			y_hat.columns = ['SBP Pred','DBP Pred']
			y_test.columns = ['SBP Test','DBP Test']
			y_hat.index = y_test.index

			ax = y_train.plot()
			y_hat.plot(ax=ax)
			y_test.plot(ax=ax)
			plt.title(f'alpha = {alpha}')
			plt.tight_layout()

	def data_checker(self, pat_n):
		plt.close('all')
		patient = self.patients[pat_n].split("'")[-2]
		print('Patient under test:',patient)
		for i in ['DBP','SBP','HR','PTT']:
			tmp = pd.read_csv('Dataset\\Patients\\'+i+'_'+patient+'.csv')
			tmp['change'] = tmp[f'x_{i}'] - tmp[f'x_{i}'].shift(1)
			interruption = list(tmp[tmp['change']<0].index)
			interruption.append(tmp.index[-1]+1)
			
			st,cnt = 0,0
			for end in interruption:
				tmp[f'x_{i}'].iloc[st:end] += 7500*cnt
				cnt += 1
				st = end
			tmp[f'x_{i}'] = tmp[f'x_{i}']/self.fs0
			tmp.index = tmp[f'x_{i}']
			tmp[i].plot(style='*-')

			if i == 'DBP': dbp = tmp
			if i == 'SBP': sbp = tmp
			if i == 'HR': hr = tmp
			if i == 'PTT': ptt = tmp

		plt.legend()
		plt.xlabel('Time [s]')
		# plt.show()
		# plt.show()
		#TODO: plot in a better way
		time_interval,time = 10,0
		TEST_PERC = .7
		max_time = max(sbp.index[-1],dbp.index[-1],hr.index[-1],ptt.index[-1])
		print(max_time)
		minimum_value_wind = 3

		while time < max_time:
			plt.close()
			tmp_dbp,tmp_sbp,tmp_hr,tmp_ptt = dbp[time:time+time_interval],sbp[time:time+time_interval],hr[time:time+time_interval],ptt[time:time+time_interval]
			l_dbp,l_sbp,l_hr,l_ptt = len(tmp_dbp),len(tmp_sbp),len(tmp_hr),len(tmp_ptt)
			difference = max(l_dbp,l_sbp,l_hr,l_ptt) - min(l_dbp,l_sbp,l_hr,l_ptt)

			#consider interval if available data > threshold
			if difference < minimum_value_wind:
				indices = []
				for x in tmp_dbp,tmp_sbp,tmp_hr,tmp_ptt:
					indices.extend(x.index)
				indices = sorted(list(dict.fromkeys(indices))) #eliminate duplicate
				df = pd.DataFrame(np.nan, index=indices, columns=['DBP','SBP','HR','PTT'])
				df['DBP'] = tmp_dbp['DBP']
				df['SBP'] = tmp_sbp['SBP']
				df['HR'] = tmp_hr['HR']
				df['PTT'] = tmp_ptt['PTT']
				df_def = df.interpolate(method='spline',order=1).dropna()
				# print(df_def)
				test_size = int(TEST_PERC*len(df_def.index))
				fig, axs = plt.subplots(3,1)
				fig.set_size_inches((16,9))

				#DBP Prediction
				X_train_dbp,y_train_dbp = df_def[['HR','PTT']].iloc[0:test_size], df_def['DBP'].iloc[0:test_size]
				X_test_dbp,y_test_dbp = df_def[['HR','PTT']].iloc[test_size::], df_def['DBP'].iloc[test_size::]
				#SBP Prediction
				X_train_sbp,y_train_sbp = df_def[['HR','PTT']].iloc[0:test_size], df_def['SBP'].iloc[0:test_size]
				X_test_sbp,y_test_sbp = df_def[['HR','PTT']].iloc[test_size::], df_def['SBP'].iloc[test_size::]
				
				axs[0].plot(y_train_dbp,label='train')
				axs[1].plot(y_train_sbp,label='train')

				maes_dbp, maes_sbp = [],[]
				alphas = [.5, 1, 2, 5, 10, 100, 1000, 10000]

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
				axs[2].set_title('Alpha')
				axs[2].set_xticks(range(len(alphas)))
				axs[2].set_xticklabels((alphas))
				axs[2].set_ylabel('MAE [-]')
				axs[2].legend()

				axs[0].plot(y_test_dbp,label='test')
				axs[0].set_ylabel('DBP [mmHg]')
				axs[0].set_xlabel('Time [s]')
				axs[0].set_title('Prediction vs. Test')
				axs[0].set_ylim(min(y_test_dbp)-10,max(y_test_dbp)+10)
				axs[0].legend(ncol=3)

				axs[1].plot(y_test_sbp,label='test')
				axs[1].set_ylabel('SBP [mmHg]')
				axs[1].set_xlabel('Time [s]')
				axs[1].set_ylim(min(y_test_sbp)-10,max(y_test_sbp)+10)
				axs[1].legend(ncol=3)
				# plt.title(f'Alpha = {alpha}')
				# ax = df.plot(style='o-')
				# df_interpolated.plot(ax=ax)
				# plt.legend()
				plt.tight_layout()
				self.create_path('Plots\\Regression')
				plt.savefig(f'Plots\\Regression\\{patient}_{round(time,1)}.png')

			time = time+time_interval


	def regression(self,y_train,X_train,X_test, alpha=1.0):
		clf = Ridge(alpha=alpha)
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)
		return pred


	def plot(self, objs, labs, xlab, ylab, title):
		plt.close('all')
		# plt.figure(figsize=())
		for obj,lab in zip(objs,labs):
			plt.plot(obj,label = lab)
		# plt.plot(ecg_int,label = 'ECG')
		plt.legend(loc = 'upper right')
		plt.ylabel(ylab)
		plt.xlabel(xlab)
		plt.title(title)

	def subplot(self, title, lst_plot, lst_lab, lst_ylab, xlab, path_name):
		fig, axs = plt.subplots(4, sharex=True, sharey=False)
		fig.suptitle(title)
		for x in range(4):
			axs[x].plot(lst_plot[x],label=lst_lab[x])
			axs[x].set_ylabel(lst_ylab[x])
			axs[x].legend('upper right')
			plt.xlabel(xlab)
			plt.savefig(path_name)
			#plt.show()
			plt.close('all')

	def create_dataset(self):
		mean, std = {'PLETH':[],'II':[],'ABP':[]}, {'PLETH':[],'II':[],'ABP':[]}
		index = np.linspace(0, 59.992, 7500)
		columns = list(mean.keys())
		PLETH 	= pd.DataFrame(index=index)
		II 		= pd.DataFrame(index=index)
		ABP 	= pd.DataFrame(index=index)
		# indices = self.abrupt_change()

		for patient in self.patients:
			patient = patient.split("'")[-2]
			print(f'########### Patient: {patient} ###########')
			patient_csv = patient+'.csv'
			file = pd.read_csv(patient_csv,sep=',')
			df = pd.DataFrame(file)
			df['change'] = df['Time'] - df['Time'].shift(1)
			split_lst = list(df.index[df['change']<0])
			df['Time'] = df['Time'].apply(lambda x: round(x,3))
			print(df)

			if float(df['Time'].max()) > 60:
				new_indices = []
				prev_value = 0
				for split_elem in split_lst:
					tmp = df[prev_value:split_elem]
					prev_value = split_elem
					new_batches = int(tmp['Time'].iloc[-1]/60+1)
					for i in range(1,int(new_batches)):
						new_indices.append(tmp.loc[(tmp['Time']>=i*60)&(tmp['Time']<(i+1)*60)].index[0])
				split_lst = sorted(split_lst + new_indices)

			#split the dataset in time slots 
			prev_value, cnt = 0, 0
			for split_elem in split_lst:
				cnt+=1
				df_split = df[prev_value:split_elem]
				df_split = df_split.drop('change',axis=1).set_index('Time')
				if max(df_split.index) > 60:
					norm = round(max(df_split.index)/60,0)
					df_split.index = [round(e-60*(norm-1),3) for e in df_split.index]
				prev_value = split_elem
				print(df_split)

				for x,y in zip([PLETH,II,ABP],['PLETH','II','ABP']):
					x[patient+'_'+str(cnt)] = df_split[y]
					mean[y].append(df_split[y].mean())
					std[y].append(df_split[y].std())

				if self.plot:
					fig, axs = plt.subplots(3,1)
					fig.set_size_inches((16,9))
					for i,j in zip(['PLETH','II','ABP'],range(3)):
						df_split[i].plot(label='reference',ax=axs[j],title=i)
					for x in ['PLETH','II','ABP']:
						a,b = x.split('-')[0].split('_')
						if int(a) == int(patient) and int(b) == int(cnt):
							[axs[2].axvline(idx,c='red') for idx in indices[x]]	
							[print(idx) for idx in indices[x]]	

					title = f'Patient {patient} - Split {cnt}'
					plt.suptitle(title)
					plt.tight_layout()
					plt.savefig('Plots\\'+title)
					# plt.show()


		if self.savefile:
			for x,y in zip([PLETH,II,ABP], ['PLETH','II','ABP']):
				x.to_csv('Dataset\\'+y+'.csv')

	def abrupt_change(self):
		df = pd.read_excel(f'Dataset\\ABP.xlsx').set_index('Unnamed: 0')
		columns = df.columns
		for col in columns:
			df[col+'-change'] =  df[col] - df[col].shift(1)
		mask = [e for e in df.columns if 'change' in e]
		# print(df[mask])

		for pat in self.patients:
			pat = pat.split("'")[-2]
			mask = [i for i in df.columns if pat in i and 'change' in i]
			plt.hist(df[mask],alpha=.7)
			# ax = df[mask].hist(bins=10)
		# print(df[mask].count())
		# plt.show()
		indices = {}
		for e in mask:
			tmp = df[e] > 20
			index = tmp.index[tmp==True]
			indices.update({e:list(index)})

		return indices

	def standardize(self): 
		#dataset analysis
		for x in ['ABP','II','PLETH']:
			df = pd.read_excel(f'Dataset\\{x}.xlsx').set_index('Unnamed: 0')
			columns = df.columns
			for col in columns:
				df[col+'-change'] =  df[col] - df[col].shift(1)
			mask = [e for e in df.columns if 'change' in e]
			print(df[mask])
			plt.hist(df[mask])
			plt.show()

			for e in mask:
				print(e)
				tmp = df[e] > 20
				index = tmp.index[tmp==True]
				# [print(x) for x in tmp]
				print(index)
			# columns = df.columns
			# index = df.index
			# tmp = df.transpose()
			# scaler = StandardScaler()
			# scaled = pd.DataFrame(scaler.fit_transform(tmp)).transpose()
			# scaled.columns = columns
			# scaled.index = index
			# scaled = scaled.interpolate()
			# print(scaled)
			# print(df)
			# plt.plot(scaled[columns[0]],label='scaled')
			# plt.plot(df[columns[0]],label='original')
			# plt.legend()
			# plt.show()	
			# exit()
		

DR = DataRegressor()
# DR.create_dataset()
# DR.interpolation()
# DR.SBP_DBP_values()

for pat_number in range(0,20):
	try:
		DR.data_checker(pat_number)
	except:
		pass


# DR.BP_retrival()
# DR.standardize()