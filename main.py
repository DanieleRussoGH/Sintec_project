#%% User: danru
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft 
import MIMICUtil as ut
import warnings
from scipy import stats
import heartpy as hp

from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.linear_model import Ridge
# from sklearn import svm
from pprint import pprint
warnings.simplefilter("ignore")
from filterpy.kalman import KalmanFilter

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

	def setup(self):
		# with open("Folder\\config.txt") as file:
		# 	config = file.read()
		# self.HeightSPmin = 0.5
		# self.HeightSPmax = 2
		self.HeightRmin = .6
		self.HeightRmax = 4

		self.CCStdCoef = 0
		self.NPEAK = 200
		self.LENSEGM = 10 
		self.K = 5
		self.fs0 = 125
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

	def cleanWindRSP(self, R, SP, dTepmin=100,dTepmax=1000):
		SP = list(SP)
		R = list(R)
		R_f = []
		SP_f = []
		""" start from the first R point """
		if SP[0] < R[0]:
			SP.pop(0)
		start = 0  
		for i in range(len(R)-1):
			found = 0
			j = start
			while found == 0 and j< len(SP):
				if SP[j] > R[i] +dTepmin and SP[j] < R[i] + dTepmax:
					found = 1
					start = j+1
					SP_f.append(SP[j])
					R_f.append(R[i])
				else:
					found = 0
					j+=1					
		return R_f, SP_f

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
		return dbp,sbp

	def SBP_DBP_values(self):
		ppg,ecg,abp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

		for col in ppg.columns:
			print(f'******************** Batch: {col} ********************')
			filt = True
			# filt = '3100038' in col
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

					DBPs_f, SBPs_f = self.cleanWindDSBP(DBPs, SBPs)
					Rs_f, SPs_f = self.cleanWindRSP(Rs, SPs)
					print('Rs:',np.array(Rs))
					print('SPs:',np.array(SPs))
					print('DBPs:',np.array(DBPs))
					print('SBPs:',np.array(SBPs))

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
					plt.savefig(f'Plots\\Batches\\{col}.png')
				
					Mptt,Mhr,Msbp,Mdbp,Mmap = self.find_ptt(SBPs_f, DBPs_f, x_SBP, x_DBP, Rs_f, SPs_f,
														index, ABP, ecg_filt, ppg_filt, col)

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

	def find_ptt(self, sbp,dbp,sbp_time,dbp_time,Rf,SPf,timestamp,abp,ecg,ppg,batch):
		""" 
			dTps : time delay between SBP point and SP point
			dTes : time delay between SBP point and R peak
			dTpe : time delay between R point and SP 
			sbp/dbp : list of SBP/DBP indexes
			sbp_time/dbp_time : timestamp corresponding to SBP/DBP
			Rf : list of R peak indexes
			SPf : list of SP indexes
			timestamp/abp/ecg/ppg : series
			patient : string """
		c = np.min([len(sbp),len(dbp)])-1
		start = 0
		Mptt,Mhr,Msbp,Mdbp,Mmap = [], [], [], [], []
		sbp_val = abp[sbp]
		dbp_val = abp[dbp]
		ctr_false = 0

		plt.close('all')
		fig, axs = plt.subplots(2,1, sharex=True) 
		fig.set_size_inches((16,9))
		axs[0].plot(dbp,abp[dbp],'r^',label = 'DBP')
		axs[0].plot(sbp,abp[sbp],'y^',label = 'SBP')
		axs[1].plot(SPf,ppg[SPf],'mx',label = 'SP')
		axs[1].plot(Rf,ecg[Rf],'gx',label = 'R')
		for i in range(c):
			sbp_t = sbp_time[i]
			#dbp_t = dbp_time[i]
			found = 0
			j = start
			ctr = 0
			while found == 0 and j < len(Rf)-1 and ctr <=20:
				ppg_time = timestamp[SPf[j]]
				ecg_time = timestamp[Rf[j]]   
				ptt = ppg_time - ecg_time	 
				# print('i = %d   j=%d'%(i,j))
				# print('ppg_t = %.2f   diff = %.2f '%(ppg_time,ppg_time-sbp_t))
				# print('ecg_t = %.2f   diff = %.2f '%(ecg_time,ecg_time-sbp_t))
				# print(ppg_time > sbp_t, ppg_time < sbp_t + 1500, ecg_time > sbp_t,ecg_time < sbp_t + 700,ppg_time < ecg_time + 800)
				if ppg_time > sbp_t and ppg_time < sbp_t + self.dTps and ecg_time > sbp_t and ecg_time < sbp_t + self.dTes and ppg_time < ecg_time + self.dTpe :
					found = 1
					start = j+1
					if j <len(Rf)-1:
						hr = 60/((timestamp[Rf[j+1]] - ecg_time)/1000)
					else:
						hr = 60/((ecg_time - timestamp[Rf[j-1]])/1000)
					Mptt.append(ptt)
					Mhr.append(hr)
					Msbp.append(sbp_val[i])
					Mdbp.append(dbp_val[i])
					Mmap.append(float((sbp_val[i]+2*dbp_val[i])/3))
					
					axs[1].vlines(Rf[j],ymin = 0, ymax=np.abs(ptt), colors='black', linestyles='dotted', linewidth=1, alpha=0.6)
					axs[1].vlines(SPf[j],ymin = 0, ymax=np.abs(ptt), colors='black', linestyles='dotted', linewidth=1, alpha=0.6)
					if i == 0 :
						axs[1].hlines(np.abs(ptt), xmin=Rf[j], xmax=SPf[j], colors='red', linestyles='solid', label='ptt')
					else:
						axs[1].hlines(np.abs(ptt), xmin=Rf[j], xmax=SPf[j], colors='red', linestyles='solid')
						
				else : 
					j = j+1
					found  = 0
					ctr = ctr + 1  
					ctr_false = ctr_false + 1
					if ctr_false == 5:
						break
		axs[0].legend(loc='upper right')
		axs[1].legend(loc='upper right')
		title = 'Corresponding values'
		fig.suptitle(title)
		plt.xlabel('Indices')
		axs[0].set_ylabel('ABP [mmHg]')
		axs[1].set_ylabel('PTT [ms]')
		plt.tight_layout()
		plt.savefig(f'Plots\\Find PTT\\{batch}_{title}.png') 
							
		return Mptt,Mhr,Msbp,Mdbp,Mmap
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
		
	def run(self):	
		filein = self.patient +'.csv'
		file1 = pd.read_csv(filein,sep=',')
		df = pd.DataFrame(file1).set_index('Time')
		print(df)

		# X,y = df[['II','PLETH']], df['ABP']
		# # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=self.RS)
		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
		# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
		# clf = Ridge(alpha=1.0)
		# clf.fit(X_train, y_train)
		# score = clf.score(X_test,y_test)
		# print(f"score is {round(score*100,1)} [%]")


		batch_size = self.LENSEGM * self.fs0
		n_batches = len(df)//batch_size

		SNRe, SNRp, SNRE, SNRP = [], [], [], []
		ECG,PPG,ABP,PTT,HR,SBP,DBP,MAP = [], [], [], [], [], [], [], []
		dREAL,dPRED,dPTTtest,dHRtest,dERRM,dERRP = [], [], [], [], [], []
		sREAL,sPRED,sPTTtest,sHRtest,sERRM,sERRP = [], [], [], [], [], []
		dREAL_nobtb, dPRED_nobtb, dPTTtest_nobtb, dHRtest_nobtb = [], [], [], []
		sREAL_nobtb, sPRED_nobtb, sPTTtest_nobtb, sHRtest_nobtb = [], [], [], []

		train_done = False
		curr_wind = 0 

		for n in range(n_batches):
			st, fin = batch_size*n,batch_size*(1+n)
			ECGs,PPGs,ABPs, = df['II'].iloc[st:fin], df['PLETH'].iloc[st:fin], df['ABP'].iloc[st:fin]
			timestamp = df.index[st:fin]

			ecg_int,ppg_int,abp_int,timestamp_int = ut.interpolate_signal(ECGs,PPGs,ABPs,timestamp,self.STEP)

			fs = self.fs0/self.STEP

			ecg_filt = ut.filter_signal(ecg_int,0.5,49,fs,'ECG')
			ppg_filt = ut.filter_signal(ppg_int,0.5,7,fs,'PPG')   

			SNRe.append(ut.noisetosignal(ecg_filt, axis = 0, ddof = 0))
			SNRp.append(ut.noisetosignal(ppg_filt, axis = 0, ddof = 0))
			SNRE.append(ut.noisetosignal(ECGs, axis = 0, ddof = 0))
			SNRP.append(ut.noisetosignal(PPGs, axis = 0, ddof = 0))

			CORR = np.correlate(ppg_filt,ppg_filt)[0] 
			nsr = ut.noisetosignal(PPGs, axis = 0, ddof = 0)

			cnt_wind = 0

			if nsr <= self.NSR_THRESHOLD and CORR >= self.CORR_THRESHOLD:
				Rs,_ = scipy.signal.find_peaks(ecg_filt,prominence=.5,width=10)
				SPs,_ = scipy.signal.find_peaks(ppg_filt,prominence=.1,width=10)
				DBPs,_ = scipy.signal.find_peaks(-abp_int,prominence=.5,width=10)
				SBPs,_ = scipy.signal.find_peaks(abp_int,prominence=.5,width=10)

				if self.plot_peaks:
					title = f'From {st} to {fin}'
					self.plot([abp_int],['ABP'],'Steps','mmHg',title)
					for x,y in zip(DBPs,SBPs):
						plt.scatter(x,abp_int[x],color='red')
						plt.scatter(y,abp_int[y],color='green')
					#plt.show()

				if train_done == False:
					if len(Rs)>=2:
						window_mean = np.mean(np.diff(Rs))
						window_std = np.std(np.diff(Rs))   
						wind_len_new = int(window_mean - window_std * self.CCStdCoef)
					else:
						window_mean = self.WindLenDefault
						window_std = 1
						wind_len_new = int(window_mean - window_std * self.CCStdCoef)

					w = [cnt_wind,1]
					arrw = [self.wind_len, wind_len_new]
					self.wind_len =  np.ma.average(arrw,weights = w)
					if self.wind_len < self.WindLenDefault-100 or self.wind_len > self.WindLenDefault + 120:
						self.wind_len = self.WindLenDefault

				dbp,sbp = ut.cleanWindDSBP(SBPs,DBPs,abp_int,self.wind_len,self.patient,self.dTsdmin,self.dTsdmax)
				if len(dbp) >= 0:
					sbp1, dbp1 = abp_int[sbp], abp_int[dbp]

					sbp_mean = np.mean(sbp1)
					sbp_std = np.std(sbp1)
					dbp_mean = np.mean(dbp1)
					dbp_std = np.std(dbp1)

					sbp_time = list(timestamp_int[sbp])
					dbp_time = list(timestamp_int[dbp])
							
					""" Clean windows with R and SP points """
					Rf,SPf = ut.cleanWindRSP(SPs,Rs,ppg_int,ecg_int,self.wind_len,self.dTepmin,self.dTepmax)
					if len(Rf)>=0:
						""" Perform PTT computtion """
						Mptt,Mhr,Msbp,Mdbp,Mmap = ut.find_ptt(self.dTps,self.dTes,self.dTpe,sbp1,dbp1,sbp_time,dbp_time,Rf,SPf,timestamp_int)

				lst_x = [ECG,PPG,ABP,PTT,HR,SBP,DBP,MAP]
				lst_y = [ecg_int,ppg_int,abp_int,Mptt,Mhr,Msbp,Mdbp,Mmap]
				for x,y in zip(lst_x,lst_y):
					x.extend(y)
				l = len(Mptt)

			cnt_wind += 1

			if len(PTT) >= self.NPEAK and train_done == False:
				print('Train')
				FO = 2
				train_done = True
				lenPTT = len(PTT)
				resid = lenPTT - self.NPEAK
				PTT_t = PTT[0:self.NPEAK]
				HR_t = HR[0:self.NPEAK]

				print('DBP')
				DBP_t = DBP[0:self.NPEAK]
				dsol,dREG_COEFF,dERRtrain,dMAEtrain,dRMSEtrain,dcurr_wind,dmean,dstd,dR2,dTTEST = ut.perform_train(FO,PTT_t, 
				HR_t, DBP_t, self.Z_SCORE_train, self.K, self.SOL_WEIGHT, curr_wind, cnt_wind)

				print('SBP')
				SBP_t = SBP[0:self.NPEAK]
				ssol,sREG_COEFF,sERRtrain,sMAEtrain,sRMSEtrain,scurr_wind,smean,sstd,sR2,sTTEST = ut.perform_train(FO,PTT_t, 
				HR_t, SBP_t, self.Z_SCORE_train, self.K, self.SOL_WEIGHT, curr_wind, cnt_wind)
				if ssol[0][0] == 0 or dsol[0][0] == 0:
					print('Error : no solution in regression')

			if train_done == True and cnt_wind > curr_wind and l>0:
				curr_wind = cnt_wind
				print('Test')
				print('DBP')
				dreal,dpred,dptttest,dhrtest,derrm,derrp = ut.perform_test(PTT,HR,DBP,dsol,FO,self.Z_SCORE_test,l,dmean,dstd)
				lst_x = [dREAL,dPRED,dPTTtest,dHRtest,dERRM,dERRP]
				lst_y = [dreal,dpred,dptttest,dhrtest,derrm,derrp]
				for x,y in zip(lst_x,lst_y):
					x.extend(y)

				print('SBP')
				sreal,spred,sptttest,shrtest,serrm,serrp = ut.perform_test(PTT,HR,SBP,ssol,FO,self.Z_SCORE_test,l,smean,sstd)
				lst_x = [sREAL,sPRED,sPTTtest,sHRtest,sERRM,sERRP]
				lst_y = [sreal,spred,sptttest,shrtest,serrm,serrp]
				for x,y in zip(lst_x,lst_y):
					x.extend(y)

				print('Test - no beat-to-beat')
				print('DBP')
				dreal_nobtb,dpred_nobtb,dptt_nobtb,dhr_nobtb = ut.perform_test_nobtb(PTT,HR,DBP,dsol,FO,l,dmean,dstd)
				lst_x = [dREAL_nobtb, dPRED_nobtb, dPTTtest_nobtb, dHRtest_nobtb]
				lst_y = [dreal_nobtb[0], dpred_nobtb[0][0], dptt_nobtb, dhr_nobtb]
				for x,y in zip(lst_x,lst_y):
					x.append(y)

				print('SBP')					   
				sreal_nobtb,spred_nobtb,sptt_nobtb,shr_nobtb = ut.perform_test_nobtb(PTT,HR,SBP,ssol,FO,l,smean,sstd)
				lst_x = [sREAL_nobtb, sPRED_nobtb, sPTTtest_nobtb, sHRtest_nobtb]
				lst_y = [sreal_nobtb[0], spred_nobtb[0][0], sptt_nobtb, shr_nobtb]
				for x,y in zip(lst_x,lst_y):
					x.append(y)


				resid = 0 #this is different from 0 only for the first cycle, when PTT is longer than NSAMPLE, so NSAMPLE are used for the training and the rest is tested
				""" Debugging part : plot if error larger than 10 """
				inn = 0

		if len(PTT) >= self.NPEAK:
			print(f'patient:{self.patient}')
			data_tmp = pd.DataFrame({'PTT':PTT,'HR':HR,'SBP':SBP,'DBP':DBP,'MAP':MAP}) 
			data_tmp = data_tmp.loc[~(data_tmp==0).all(axis=1)]
			data_tmp.dropna(inplace = True)
			data_tmp = data_tmp[(np.abs(stats.zscore(data_tmp)) < self.Z_SCORE_test).all(axis=1)]
			print('\nPTT_mean = %.2f  PTT_std = %.2f' %(data_tmp['PTT'].mean(), data_tmp['PTT'].std()))
			print('DBP')
			print('Regression coeff : a_ptt = %.3f   b_hr = %.3f' %(dsol[0][0],dsol[0][1]))
			print('RMSE_train mean = %.2f   RMSE std = %.2f' %(np.mean(dRMSEtrain), np.std(dRMSEtrain)))
			print('MAE_train mean = %.2f   MAE std = %.2f' %(np.mean(dMAEtrain), np.std(dMAEtrain)))
			#print('R2mean = %.3f   Ttest : statistic = %.3f   p-value = %.3f' %(np.mean(R2),TTEST[0][0],TTEST[1][0]))#careful! see all values!
			print('MAE_test = %.2f  RMSE_test = %.2f   NsamplesTest = %d \n' %(mean_absolute_error(dREAL,dPRED), mean_squared_error(dREAL,dPRED,squared = False),len(dREAL)))

			err_bp,err_ptt,right_bp,right_ptt = [], [], [], []
			err_perc = 0

			err = np.array(dREAL) - np.array(dPRED)
			for i in range(len(err)):
				if np.abs(err[i]) >= 10 :
					err_bp.append(dREAL[i])
					err_ptt.append(dPTTtest[i])
				if np.abs(err[i]) < 10:
					right_bp.append(dREAL[i])
					right_ptt.append(dPTTtest[i])
			err_perc = len(err_ptt) / len(err) * 100
			print('Error >= 10 mmHg occurs the %.2f %% of the times' %(err_perc))
			print('Number of samples tested = %d   err>10 occurrences = %d'%(len(dREAL),len(err_ptt)))
			print('Wrong prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f' %(np.mean(err_ptt),np.std(err_ptt),np.mean(err_bp),np.std(err_bp)))
			print('Right prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f \n' %(np.mean(right_ptt),np.std(right_ptt),np.mean(right_bp),np.std(right_bp)))

			x = np.arange(0,len(dPRED),1)
			title = 'DBP Real vs. Pred - test'
			self.plot([dREAL,dPRED],['Real','Pred'],'Test samples','BP [mmHg]',title)
			plt.savefig(self.patient + '_' +'RealVsPredFinDBP.png',format='png')
			#plt.show()
			plt.close('all')

			plt.hist(np.array(dREAL)- np.array(dPRED),bins=50)
			plt.xlabel('Error real-pred')
			plt.ylabel('Instances')
			plt.title('DBP RealVsPred_test_hist')
			plt.savefig(self.patient + '_' + title + 'DBP.png',format='png')
			#plt.show()
			plt.close('all')

			title = 'DBP and error test'
			lst_plot = [dREAL,dPTTtest,dHRtest,np.array(dREAL)- np.array(dPRED)]
			lst_lab = ['BP','ptt','hr','error']
			lst_ylab = ['BP [mmHg]','PTT [ms]','HR [bpm]','ERR']
			path_name = self.patient + '_' +title + '.png'
			self.subplot(title,lst_plot,lst_lab,lst_ylab,'Test samples',path_name)

			x = np.arange(0,len(dPRED_nobtb),1)
			title = 'DBP Real vs pred test - no beat to beat'
			self.plot([dREAL_nobtb,dPRED_nobtb],['Real','Pred'],'Test samples','BP [mmHg]',title)
			plt.savefig(self.patient + '_' +'RealVsPredFinNobtbDBP.png',format='png')
			#plt.show()
			plt.close('all')

			title = 'DBP and error test - no beat to beat'
			err_nobtb = np.array(dREAL_nobtb)- np.array(dPRED_nobtb)
			lst_plot = [dREAL_nobtb,dPTTtest_nobtb,dHRtest_nobtb,err_nobtb]
			lst_lab = ['BP','ptt','hr','error']
			lst_ylab = ['BP [mmHg]','PTT [ms]','HR [bpm]','ERR']
			path_name = self.patient + '_' +title + '.png'
			self.subplot(title,lst_plot,lst_lab,lst_ylab,'Test samples',path_name)

			err_bp_nobtb, err_ptt_nobtb, right_bp_nobtb, right_ptt_nobtb = [], [], [], []
			err_perc_nobtb = 0
			for i in range(len(err_nobtb)):
				if np.abs(err_nobtb[i]) >= 10 :
					err_bp_nobtb.append(dREAL_nobtb[i])
					err_ptt_nobtb.append(dPTTtest_nobtb[i])
				if np.abs(err_nobtb[i]) < 10:
					right_bp_nobtb.append(dREAL_nobtb[i])
					right_ptt_nobtb.append(dPTTtest_nobtb[i])
			err_perc_nobtb = len(err_ptt_nobtb) / len(err_nobtb) * 100

			print('Test no beat-to-beat : intervals of %d seconds' %(self.LENSEGM))
			print('MAE_test = %.2f  RMSE_test = %.2f   NsamplesTest = %d \n' %(mean_absolute_error(dREAL_nobtb,dPRED_nobtb), mean_squared_error(dREAL_nobtb,dPRED_nobtb,squared = False),len(dREAL_nobtb)))
			print('Error >= 10 mmHg occurs the %.2f %% of the times' %(err_perc_nobtb))
			print('Number of samples tested = %d   err>10 occurrences = %d'%(len(dREAL_nobtb),len(err_ptt_nobtb)))
			print('Wrong prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f' %(np.mean(err_ptt_nobtb),np.std(err_ptt_nobtb),np.mean(err_bp_nobtb),np.std(err_bp_nobtb)))
			print('Right prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f \n' %(np.mean(right_ptt_nobtb),np.std(right_ptt_nobtb),np.mean(right_bp_nobtb),np.std(right_bp_nobtb)))

			print('SBP')
			print('Regression coeff : a_ptt = %.3f   b_hr = %.3f' %(ssol[0][0],ssol[0][1]))
			print('RMSE_train mean = %.2f   RMSE std = %.2f' %(np.mean(sRMSEtrain), np.std(sRMSEtrain)))
			print('MAE_train mean = %.2f   MAE std = %.2f' %(np.mean(sMAEtrain), np.std(sMAEtrain)))
			#print('R2mean = %.3f   Ttest : statistic = %.3f   p-value = %.3f' %(np.mean(R2),TTEST[0][0],TTEST[1][0]))#careful! see all values!
			print('MAE_test = %.2f  RMSE_test = %.2f   NsamplesTest = %d \n' %(mean_absolute_error(sREAL,sPRED), mean_squared_error(sREAL,sPRED,squared = False),len(sREAL)))

			err_bp, err_ptt, right_bp, right_ptt = [], [], [], []
			err_perc = 0

			err = np.array(sREAL)- np.array(sPRED)
			for i in range(len(err)):
				if np.abs(err[i]) >= 10 :
					err_bp.append(sREAL[i])
					err_ptt.append(sPTTtest[i])
				if np.abs(err[i]) < 10:
					right_bp.append(sREAL[i])
					right_ptt.append(sPTTtest[i])
			err_perc = len(err_ptt) / len(err) * 100
			print('Error >= 10 mmHg occurs the %.2f %% of the times' %(err_perc))
			print('Number of samples tested = %d   err>10 occurrences = %d'%(len(sREAL),len(err_ptt)))
			print('Wrong prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f' %(np.mean(err_ptt),np.std(err_ptt),np.mean(err_bp),np.std(err_bp)))
			print('Right prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f \n' %(np.mean(right_ptt),np.std(right_ptt),np.mean(right_bp),np.std(right_bp)))

			x = np.arange(0,len(dPRED),1)
			title = 'SBP Real vs. Pred - test'
			self.plot([sREAL,sPRED],['Real','Pred'],'Test samples','BP [mmHg]',title)
			plt.savefig(self.patient + '_' +'RealVsPredFinSBP.png',format='png')
			#plt.show()
			plt.close('all')

			title = 'SBP RealVsPred_test_hist'
			plt.hist(np.array(sREAL)- np.array(sPRED),bins=50)
			plt.xlabel('Error real-pred')
			plt.ylabel('Instances')
			plt.title(title)
			plt.savefig(self.patient + '_' + title + 'SBP.png',format='png')
			#plt.show()
			plt.close('all')

			title = 'SBP and error test'
			lst_plot = [sREAL,sPTTtest,sHRtest,np.array(sREAL)- np.array(sPRED)]
			lst_lab = ['BP','ptt','hr','error']
			lst_ylab = ['BP [mmHg]','PTT [ms]','HR [bpm]','ERR']
			path_name = self.patient + '_' +title + '.png'
			self.subplot(title,lst_plot,lst_lab,lst_ylab,'Test samples',path_name)

			x = np.arange(0,len(sPRED_nobtb),1)
			title = 'SBP: Real vs pred- Test (no beat to beat)'
			self.plot([sREAL_nobtb,sPRED_nobtb],['Real','Pred'],'Test samples','BP [mmHg]',title)
			plt.savefig(self.patient + '_' +'RealVsPredFinNobtbSBP.png.png',format='png')
			#plt.show()
			plt.close('all')

			title = 'SBP and error test - no beat to beat'
			lst_plot = [sREAL_nobtb,sPTTtest_nobtb,sHRtest_nobtb,err_nobtb]
			lst_lab = ['BP','ptt','hr','error']
			lst_ylab = ['BP [mmHg]','PTT [ms]','HR [bpm]','ERR']
			path_name = self.patient + '_' +title + '.png'
			self.subplot(title,lst_plot,lst_lab,lst_ylab,'Test samples',path_name)

			err_bp_nobtb, err_ptt_nobtb, right_bp_nobtb, right_ptt_nobtb = [], [], [], []
			err_perc_nobtb = 0

			for i in range(len(err_nobtb)):
				if np.abs(err_nobtb[i]) >= 10 :
					err_bp_nobtb.append(sREAL_nobtb[i])
					err_ptt_nobtb.append(sPTTtest_nobtb[i])
				if np.abs(err_nobtb[i]) < 10:
					right_bp_nobtb.append(sREAL_nobtb[i])
					right_ptt_nobtb.append(sPTTtest_nobtb[i])
			err_perc_nobtb = len(err_ptt_nobtb) / len(err_nobtb) * 100
			print('Test no beat-to-beat : intervals of %d seconds' %(self.LENSEGM))
			print('MAE_test = %.2f  RMSE_test = %.2f   NsamplesTest = %d \n' %(mean_absolute_error(sREAL_nobtb,sPRED_nobtb), mean_squared_error(sREAL_nobtb,sPRED_nobtb,squared = False),len(sREAL_nobtb)))
			print('Error >= 10 mmHg occurs the %.2f %% of the times' %(err_perc_nobtb))
			print('Number of samples tested = %d   err>10 occurrences = %d'%(len(sREAL_nobtb),len(err_ptt_nobtb)))
			print('Wrong prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f' %(np.mean(err_ptt_nobtb),np.std(err_ptt_nobtb),np.mean(err_bp_nobtb),np.std(err_bp_nobtb)))
			print('Right prediction :\nPTT mean = %.2f	std = %.2f   BP mean = %.2f   std = %.2f \n' %(np.mean(right_ptt_nobtb),np.std(right_ptt_nobtb),np.mean(right_bp_nobtb),np.std(right_bp_nobtb)))
	   
DR = DataRegressor()
# DR.create_dataset()
# DR.interpolation()
DR.SBP_DBP_values()
# DR.standardize()