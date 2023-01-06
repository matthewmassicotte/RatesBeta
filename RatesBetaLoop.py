#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:09:41 2022

@author: matthewmassicotte
"""


#TODO:
#    FIX WEEKLY RETURNS
#    FIX MONTHLY WEIGTHED RETURNS
if __name__ == '__main__':

    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import copy
    import wrds
    import multiprocess as mp
    import datetime
    
    db = wrds.Connection(wrds_username='mattmass')
    
#%%   Define functions that will be used for parrallelizing the slow portions of the code (using multiprocessing pools)

#This is used to get weekly returns.  It sets a common weekly index and calculates the returns, rate chagnes.  each permno (ticker) is a seperate process
#NEED TO FIX WEEKLY RETURNS
def makeWeekly(args):
    df,indexBase = args
    df= pd.merge(indexBase, df, how='inner', on=['date'])
    # df['ret'] = df['prc'].pct_change()
    df['ret'] = df['weeklyret']
    df['rate'] = df['rate'].diff() 
    return df

#This runs the rates beta regressions.  Each year is a seperate process, which is then grouped at the ticker level for regression
def getBetas(args):
    year,one_df=args
    betaDict2 = dict()
    for ticker, two_df in one_df.groupby(['permno']):
        Y = two_df['ret']
        X = two_df['rate']
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        betaDict2[ticker] = results.params['rate']
        # print(two_df)
    print(year)
    return betaDict2

#this creates the decile level portfolios.   Each year is a seperate process, it is then grouped by year, then decile.  data is aggregated using a marketcap weighted average.
def calcWeightedReturns(args):
    parts = []
    aggParts=[]
    year,one_df,betaDictCopy=args

    if year not in betaDictCopy.keys():
        print(str(year)+" not in betadict")
        return -1;

    one_df=one_df.set_index(['permno'])
    one_df['shrout*prc']=one_df['shrout'].multiply(one_df['prc'])    
    
    for decile, two_df in one_df.groupby(by=betaDictCopy[year]):
        two_df['Decile'] = decile
        
        for date, three_df in two_df.groupby(['date']):
            three_df['weight']=three_df['shrout*prc'].divide(three_df['shrout*prc'].sum())
            three_df['weightedReturn']=three_df['weight'].dot(three_df['ret'])
            three_df['weightedEY']=three_df['weight'].dot(three_df['EY'])
            three_df['weightedPriceBook']=three_df['weight'].dot(three_df['PriceBook'])
            three_df['weightedDY']=three_df['weight'].dot(three_df['DY'])
            three_df=three_df.reset_index()
            three_df=three_df.set_index(['date'])
            parts.append(three_df)                                                          
            aggParts.append(pd.DataFrame([date,decile,three_df['rate'][0],three_df['weightedReturn'][0],three_df['weightedPriceBook'][0],three_df['weightedEY'][0],three_df['weightedDY'][0]]).T)
    print(year)
    return [parts,aggParts]
    
#%% Get WRDS data and merge (using CCM)

if __name__ == '__main__':
    
    startDate = '1/01/1963'
    linkstartDate = '1/01/1900'

    #this will be used to remove any companies that were delisted within a year
    delists = db.raw_sql("""select delistingdt, permno
                            from crsp.stkdelists 
                            where delistingdt>= '""" +startDate+ """'
                            """, 
                          date_cols=['delistingdt'])
    
    delists['year']=delists['delistingdt'].astype('datetime64[ns]').dt.year

    #this is used to create the weekly baseline index, and for market betas
    sp500 = db.raw_sql("""select caldt, vwretd
                            from crsp.dsp500
                            where caldt>='""" +startDate+ """'
                            """, 
                          date_cols=['caldt'])
  
    #used to create an index for weekly returns
    sp500['caldt']=sp500['caldt'].astype('datetime64[ns]')
    
    #this is used for CAPM Betas
    sp500_monthly = db.raw_sql("""select caldt, vwretd
                            from crsp.msp500
                            where caldt>='""" +startDate+ """'
                            """, 
                          date_cols=['caldt'])
  
    #used to create an index for weekly returns
    sp500_monthly['caldt']=sp500_monthly['caldt'].astype('datetime64[ns]')

    #this gets the earnings, dividend etc data.
    compustat = db.raw_sql("""select gvkey, datadate, fyear, BKVLPS, EPSPX, DVC, PRCC_F, CSHO
                            from comp.funda 
                            where datadate>='""" +linkstartDate+ """'
                            and indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
                            """, 
                          date_cols=['datadate'])
    #add ratios
    compustat['EY'] = compustat['epspx'].divide(compustat['prcc_f'])
    compustat['PriceBook'] = compustat['prcc_f'].divide(compustat['bkvlps'])
    compustat['DY'] = (compustat['dvc'].divide(compustat['csho'])).divide(compustat['prcc_f'])
    compustat['datadate']=compustat['datadate'].astype('datetime64[ns]')
    compustat = compustat.rename(columns={'fyear': 'year'})

    #stock level daily data
    crsp = db.raw_sql("""select permno, date, prc, ret, shrout 
                            from crsp.dsf 
                            where date>='""" +startDate+ """'
                            """, 
                          date_cols=['date'])
    
    crsp['date']=crsp['date'].astype('datetime64[ns]')
    crsp['year']=crsp['date'].dt.year
    # crsp['dayOfWeek']=crsp['date'].dt.day_name()
    crsp['prc']=crsp['prc'].abs()
    crsp=crsp[crsp['prc']>5]
    
    crsp['weeklyret']=crsp['ret']+1
    crsp['weeklyret'] = (crsp
      .set_index("date")
      .groupby(["permno",pd.Grouper(freq='W-WED')])["weeklyret"].transform('prod')).values-1

    # crsp2=crsp[crsp["permno"]==10180]
    

    #CCM link table used to merge compustat and CRSP
    link = db.raw_sql("""select *
                            from crsp.ccmxpf_linktable 
                            where linkenddt >='""" +linkstartDate+ """'
                            and USEDFLAG =1
                           """)
    link = link[link['linktype'].isin(['LU','LC'])]
    link = link[link['linkprim'].isin(['P','C','J'])]
    link['linkenddt'] = link['linkenddt'].fillna(value=datetime.date.today())
    link['linkenddt']=link['linkenddt'].astype('datetime64[ns]')
    link['endyear']=link['linkenddt'].dt.year
    link['linkdt'] = link['linkdt'].fillna(value=datetime.date.today())
    link['linkdt']=link['linkdt'].astype('datetime64[ns]')
    link['startyear']=link['linkdt'].dt.year
    link = link.rename(columns={'lpermno': 'permno'})

    merge= pd.merge(compustat, link, how='inner', on=['gvkey'])
    merge=merge[(merge["year"]>=merge["startyear"]) & (merge["year"]<=merge["endyear"])]

    data= pd.merge(crsp, merge, how='inner', on=['permno','year'])
    
    #  save data to local file
    # data.to_csv('./test_data.csv')
    
    #  Load data from local file
    # data = pd.read_csv('/Users/matthewmassicotte/Documents/test_data.csv')
    # data = data.iloc[: , 1:]


#%%# Get rates data from FRED
if __name__ == '__main__': 
    
    
        # WEEKLY OR DAILY REturns
        # weekly = True
        
    from fredapi import Fred
    fred = Fred(api_key='32e1507c555edb8d7de8a0977f22841d')
    import pandas as pd
        
    x=[[False,'DGS10'],[False,'DGS2'],[False,'DTB3'],[True,'DGS10'],[True,'DGS2'],[True,'DTB3']]
        
    for args in x:
        weekly,ratecode=args


        
        # ratecode = 'DGS10'
        if ratecode == 'DGS2':
            rateType='2yr'
        elif ratecode == 'DGS10':
            rateType='10yr'
        elif ratecode == 'DTB3':
            rateType='3M'  #DGS3MO
            
        print(weekly)
        print(rateType)
        print()
                
        #get rates from fred
        if weekly:
            rates = fred.get_series(ratecode,observation_start=startDate)#.diff()
        else:
            rates = fred.get_series(ratecode,observation_start=startDate).diff()
            
        rates.name="rate"
        rates=rates.to_frame()
        data.index=data['date']
        data.index = data.index.astype('datetime64[ns]')
        rates.index = rates.index.astype('datetime64[ns]')
        data.index = data.index.round(freq ='D')
        rates.index = rates.index.round(freq ='D')
        
        # Merge CRSP and Rates data (inner join) - only common dates remain
        merged = data.merge(rates,left_index=True,right_index=True).dropna()
        #RETURNS FOR MONTHY AND WEEKLY ARE PROBS WRONG
        #run pool to make return horizon weekly (if weekly flag is true)
        if weekly:
            print("running weekly pool")
            sp500.index=sp500['caldt']
            indexBase = sp500.resample("W-WED").last()
            indexBase.columns=["date",'vwret']
            parts = []
            pool = mp.Pool(mp.cpu_count())
            ret_list = pool.map(makeWeekly, [[df,indexBase] for permno, df in merged.groupby(['permno'])])
            pool.close()
            pool.join()
            merged=pd.concat(ret_list).dropna()
            merged.index=merged["date"]
            print("done with pool")
    
        # removes any delisted stocks from the year in which they were delisted
        merged = pd.merge(merged, delists, on=['permno','year'], how="outer", indicator=True
                      ).query('_merge=="left_only"')
        
        #drop columns that have no more utility 
        merged=merged.drop(['delistingdt', '_merge','gvkey','bkvlps','startyear','datadate','epspx',	'dvc',	'prcc_f',	'csho','linkprim',	'liid'	,'linktype',	'lpermco',	'usedflag',	'linkdt',	'linkenddt'	,'endyear'], axis=1)
        merged.index=merged['date']
    # #%%
    
    # crsp2=crsp[crsp['year'].isin([1986,1987,1988])]
    # #%%
    # merged2=merged[merged['year'].isin([1986,1987,1988])]
    
    # #%% running multiprocessing pools
    # if __name__ == '__main__':
        print('running multiprocessing pools')
        
        #trim data for testing (so it is faster)
        test=merged#.iloc[-5000000:,:]
        
        # Caluclate beta for each permno(ticker) for each year 
        betaDict = dict()
        
        #this creates a multiprocess pool where each core will seperately calc betas for a given year
        pool = mp.Pool(mp.cpu_count())
        ret_list = pool.map(getBetas, [[year,one_df] for year, one_df in test.groupby(test.index.year)])
        pool.close()
        pool.join()
        
        # convert list of return values (dictionaries for each year) into a dictionary of dictionaries
        #Test(dataset) is in chronological order, so the grouping will be too (map preserves order)
        for i,year in enumerate(test.index.year.unique()):
            betaDict[year] = ret_list[i]
    
        #   Creates a dictionary mapping the permno(ticker) to a decile for each year
        #deciles are calculated using the previous years betas
        #betaDict is a dictionary of dictionaries.  For each year there is a dictionary mapping each permno to its beta
    
        betaDictCopy = copy.deepcopy(betaDict)
        #Convert each year's dict to a dataframe to use qcut (to assign deciles) 
        for year in sorted(list(betaDictCopy.keys())): 
            betaDictCopy[year]=pd.DataFrame.from_dict(betaDictCopy[year], orient ='index')
            betaDictCopy[year].columns=['Beta']
            betaDictCopy[year]=betaDictCopy[year][betaDictCopy[year]['Beta']!=0]
            
            #if previous years data exists, assign deciles based on that years data
            if year-1 in betaDict.keys():
                betas=betaDictCopy[year-1]['Beta'].squeeze()
                #this preserves the index, any indeces that werent in year-1 will be nan (later dropped), any in year-1 but not year not included
                betaDictCopy[year]['Decile'] = pd.qcut(betas, 10,labels = [1,2,3,4,5,6,7,8,9,10])
        
                #removes any years where prior year data doesnt exist (prop year determines decile) 
                if year-2 not in betaDict.keys():
                    betaDictCopy.pop(year-1)
                    
                    
        #convert back to dictionary of dictionaries (keeps only the decile column)
        for year in list(betaDictCopy.keys()):
            betaDictCopy[year] = betaDictCopy[year].iloc[: , 1:].squeeze().dropna().to_dict()
    

        # Calculate the weighted returns for each decile (using multiprocess pool)
        pool = mp.Pool(mp.cpu_count())
        ret_list = pool.map(calcWeightedReturns, [[year, one_df,betaDictCopy] for year, one_df in test.groupby(test.index.year)])
        pool.close()
        pool.join()
     
        result = []
        
        aggregate=[]
    
        #turn return lists into dataframes
        #Aggregate is at decile portfolio level
        #result is permno level
        for ret in ret_list:
            if ret !=-1:
                p,ap = ret
                if len(p)==0:
                    print('issue with one output - skipping - likely lack of data')
                    continue
                if type(result)==list:
                    result=pd.concat(p)
                    aggregate=pd.concat(ap)
                else:
                    result=pd.concat([result,pd.concat(p)])
                    aggregate=pd.concat([aggregate,pd.concat(ap)])
    
        aggregate.columns=['date','decile','rate','weightedReturn','weightedPriceBook',	'weightedEY',	'weightedDY']
        aggregate=aggregate.set_index(['date'])

        # Calculate betas for the deciles (weighted retun regressed on rates)
        aggregate['beta']=-1
        parts = []
        aggregate.index = aggregate.index.astype('datetime64[ns]')
    
        for year, one_df in aggregate.groupby(aggregate.index.year):
            for decile, two_df in one_df.groupby(['decile']):
                Y = two_df['weightedReturn']
                X = two_df['rate']
                X = sm.add_constant(X)
                Y = Y.astype('float')
                X = X.astype('float')
                model = sm.OLS(Y,X)
                results = model.fit()
                
                two_df['weightedPriceBook']=two_df['weightedPriceBook'].mean()
                two_df['weightedEY']=two_df['weightedEY'].mean()
                two_df['weightedDY']=two_df['weightedDY'].mean()
                
                two_df['beta']=results.params['rate']
                parts.append(two_df)
        aggregate=pd.concat(parts)
        


        spread=(aggregate[aggregate['decile']==1]-aggregate[aggregate['decile']==10]).dropna()
        spread['decile']=-1
        spread['rate']=aggregate[aggregate['decile']==10]['rate']
        
        aggregate=pd.concat([aggregate,spread])
        
        years= aggregate[aggregate['decile']==1].index.year.unique()
        
        spData = pd.read_excel("/Users/matthewmassicotte/Documents/GitHub/RatesBeta/spData.xlsx")
        
        spData=spData[spData['Year'].isin(years)]
        
        
        
        spEY=spData['Earnings Yield'].mean()
        spDY=spData['Dividend Yield'].mean()

        sp500_rate= sp500.merge(rates,left_index=True,right_index=True).dropna()
        
        Y = sp500_rate['vwretd']
        X = sp500_rate['rate']
        X = sm.add_constant(X)
        Y = Y.astype('float')
        X = X.astype('float')
        model = sm.OLS(Y,X)
        results = model.fit()
        spRateBeta=results.params['rate']
        spCapmBeta=1


        #Full-sample EY,DY,PB, Rates Betas
        parts = [] 
        for decile, two_df in aggregate.groupby(['decile']):
                Y = two_df['weightedReturn']
                X = two_df['rate']
                X = sm.add_constant(X)
                Y = Y.astype('float')
                X = X.astype('float')
                model = sm.OLS(Y,X)
                results = model.fit()
                
                pb=two_df['weightedPriceBook'].mean()
                ey=two_df['weightedEY'].mean()
                dy=two_df['weightedDY'].mean()
                
                beta=results.params['rate']
                parts.append([beta,pb,ey,dy,decile])
        # aggregate=pd.concat(parts)
        aggregate_fs=pd.DataFrame.from_records(parts)
        aggregate_fs.columns=['RateBeta','PriceBook','EarningsYeild','DividendYeild','decile']
        aggregate_fs.index=aggregate_fs['decile'].astype(int)
        aggregate_fs=aggregate_fs.drop(['decile'],axis=1)
        
    
        
       # # %% CAPM Betas then plots
        print('CAPM Betas then plots')
    # if __name__ == '__main__':
        #CAPM Betas merged aggregate with sp500 and 1m rates data, then do capm form regression using monthly returns
        capmParts=[]
        db = wrds.Connection(wrds_username='mattmass')
        #Merge FRED and CRSP 1mo rates (both have missing time periods)
        mrate=fred.get_series('DGS1MO',observation_start='1963-1-1').dropna()
        mrate.index = mrate.index.astype('datetime64[ns]')
        mrate.name="mrate"
        mrate=mrate.to_frame()/100
        mrate['date']=mrate.index
        crsprate = db.raw_sql("""select ave_1, qdate
                                from crsp.riskfree 
                                where qdate>='1/01/1963'""", 
                              date_cols=['qdate'])
        crsprate['ave_1']=    crsprate['ave_1']/100
        crsprate['date']=crsprate['qdate'].astype('datetime64[ns]')
        ratemerged = pd.merge(mrate, crsprate, on=['date'],how="outer")
        ratemerged.index=ratemerged['date']
        ratemerged=ratemerged.sort_index()
        x=ratemerged['mrate'].fillna(ratemerged['ave_1'])
    
    
        sp500_monthly.index=sp500_monthly['caldt']
        merge=sp500_monthly.merge(x,left_index=True,right_index=True).dropna()
        
        
        
        aggregate['cumweightedReturn']=aggregate['weightedReturn']+1
        monthlyAgg = (aggregate
                      .groupby(["decile",pd.Grouper(freq='M')])['cumweightedReturn']
                      .prod()-1).reset_index(level='decile')
        
        
        
        capmMerged = merge.merge(monthlyAgg,left_index=True,right_index=True).dropna()
    
        capmMerged['x']=capmMerged['vwretd'].sub(capmMerged['mrate'])
        capmMerged['y']=capmMerged['cumweightedReturn'].sub(capmMerged['mrate'])
        capmMerged=capmMerged.dropna()

        #performs market beta regression for each year
        parts=[]
        capmBetaDict=dict()
        for year, one_df in capmMerged.groupby(capmMerged.index.year):
            capmBetaDict2=dict()
            for decile, two_df in one_df.groupby(['decile']):
                
                Y = two_df['y']
                X = two_df['x']
                X = sm.add_constant(X)
                Y = Y.astype('float')
                X = X.astype('float')
                model = sm.OLS(Y,X)
                results = model.fit()
                two_df['capmbeta']=results.params['x']
                
                parts.append(two_df)
                capmBetaDict2[decile]=results.params['x']
            capmBetaDict[year]=capmBetaDict2
        capmResults=pd.concat(parts)
        
        #Full sample campbeta
        parts=[]
        for decile, two_df in capmMerged.groupby(['decile']):
            Y = two_df['y']
            X = two_df['x']
            X = sm.add_constant(X)
            Y = Y.astype('float')
            X = X.astype('float')
            model = sm.OLS(Y,X)
            results = model.fit()
            # two_df['capmbeta']=results.params['x']
            parts.append([decile,results.params['x']])
        capmResults_fs=pd.DataFrame.from_records(parts)
        aggregate_fs['CapmBeta']=capmResults_fs[1].values

        if weekly:
            freq='weeklyReturns'
        else:
            freq='dailyReturns'

        #Plot capmbetas over time
        plt.rcParams['figure.figsize'] = (10,10)
        for decile, one_df in capmResults[capmResults['decile']>0].groupby(['decile']):   
            one_df.name=decile
            plt.plot(one_df.index,one_df['capmbeta'])
        plt.legend([1,2,3,4,5,6,7,8,9,10])
        # plt.ylim([-.25,3])
        plt.title(rateType+" CAPM Beta")
        plt.savefig('/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Capm-Beta.png')
        plt.show() 
    
        
        parts=[]
        #Add to 'aggregate' so it saves with with rest of the data
        for year, one_df in aggregate[aggregate['decile']>0].groupby(aggregate[aggregate['decile']>0].index.year):
            for decile, two_df in one_df.groupby(['decile']):
                two_df['capmBeta'] = capmBetaDict[year][decile]
                parts.append(two_df)
        aggregate=pd.concat(parts)
                
    
        #Plot betas over time
        plt.rcParams['figure.figsize'] = (10,10)
        for decile, one_df in aggregate[aggregate['decile']>0].groupby(['decile']):   
            one_df.name=decile
            plt.plot(one_df.index,one_df['beta'])
        plt.legend([1,2,3,4,5,6,7,8,9,10])
        plt.ylim([-.2,.5])
        plt.title(rateType+" Rate Beta")
        plt.savefig('/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Rate-Beta.png')
        plt.show()    
    
    
        
        #Plot price-book ratio over time
        plt.rcParams['figure.figsize'] = (10,10)
        for decile, one_df in aggregate[aggregate['decile']>0].groupby(['decile']):   
            one_df.name=decile
            plt.plot(one_df.index,one_df['weightedPriceBook'])
        plt.ylim([-10,30])
        plt.legend([1,2,3,4,5,6,7,8,9,10])
        plt.title(rateType+" Price/Book")
        plt.savefig('/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Price-Book.png')
        plt.show()
    
        
        #Plot earning yield over time
        plt.rcParams['figure.figsize'] = (10,10)
        for decile, one_df in aggregate[aggregate['decile']>0].groupby(['decile']):   
            one_df.name=decile
            plt.plot(one_df.index,one_df['weightedEY'])
        plt.ylim([-.4,.3])
        plt.legend([1,2,3,4,5,6,7,8,9,10])
        plt.title(rateType+" Earnings Yield")
        plt.savefig('/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Earnings-Yield.png')
        plt.show()
    
        
        #Plot div yield over time
        plt.rcParams['figure.figsize'] = (10,10)
        for decile, one_df in aggregate[aggregate['decile']>0].groupby(['decile']):   
            one_df.name=decile
            plt.plot(one_df.index,one_df['weightedDY'])
        plt.ylim([-.01,.1])
        plt.legend([1,2,3,4,5,6,7,8,9,10])
        plt.title(rateType+" Dividend Yield")
        plt.savefig('/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Dividend-Yield.png')
        plt.show()
    
        fig, ax = plt.subplots()
        qty = result.groupby(['year'])['permno'].count()
        ax.plot(qty.index,qty)
        ax.set_title('datapoints/year')
        # plt.savefig('./'+rateType+'-DataPoint-Year.png')
        plt.savefig('/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-DataPoint-Year.png')
        # ax.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
        plt.show()
    #     #%%
    # if __name__ == '__main__':
        #save to file

        emptyDF= pd.DataFrame()
    
        with pd.ExcelWriter("/Users/matthewmassicotte/Documents/GitHub/RatesBeta/SummaryStats-"+rateType+"-"+freq+".xlsx") as writer:
            emptyDF.to_excel(writer, sheet_name='charts', index=False)
            worksheet = writer.sheets['charts']
            worksheet.insert_image('C2','/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Rate-Beta.png')
            worksheet.insert_image('C50','/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Capm-Beta.png')
            worksheet.insert_image('C100','/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Dividend-Yield.png')
            worksheet.insert_image('C150','/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Earnings-Yield.png')
            worksheet.insert_image('C200','/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-Price-Book.png')
            worksheet.insert_image('C250','/Users/matthewmassicotte/Documents/GitHub/RatesBeta/Plots/'+rateType+"-"+freq+'-DataPoint-Year.png')
            
            aggregate_fs[aggregate_fs.index>0].to_excel(writer, sheet_name=rateType+"-"+freq, index=True)
            writer.sheets[rateType+"-"+freq].set_column('A:J', 15)
            writer.sheets[rateType+"-"+freq].conditional_format('B2:B11', {'type': '3_color_scale'})
            writer.sheets[rateType+"-"+freq].conditional_format('C2:C11', {'type': '3_color_scale'})
            writer.sheets[rateType+"-"+freq].conditional_format('D2:D11', {'type': '3_color_scale'})
            writer.sheets[rateType+"-"+freq].conditional_format('E2:E11', {'type': '3_color_scale'})
            writer.sheets[rateType+"-"+freq].conditional_format('F2:F11', {'type': '3_color_scale'})
            
            df=aggregate_fs[aggregate_fs.index<0]
            df.index=['Spread']
            df.to_excel(writer, sheet_name=rateType+"-"+freq, index=True,startrow= 13,header=True)
            df=pd.DataFrame([spRateBeta,'NA', spEY,spDY,spCapmBeta]).T
            df.index=['Market']
            df.to_excel(writer, sheet_name=rateType+"-"+freq,startrow= 15,header=False,index=True)


    
    
            for decile, one_df in aggregate.groupby(['decile']):   
                one_df.to_excel(writer, sheet_name=str(int(decile)), index=True)
                worksheet = writer.sheets[str(int(decile))]
                worksheet.set_column('A:J', 20)
            
        writer.close()

        # try:
        #     with pd.ExcelWriter("/Users/matthewmassicotte/Documents/GitHub/RatesBeta/FullSampleStats.xlsx",engine='openpyxl',mode='a') as writer:
        #         aggregate_fs[aggregate_fs.index>0].to_excel(writer, sheet_name=rateType+"-"+freq, index=True)
        #         writer.sheets[rateType+"-"+freq].set_column('A:J', 15)
        #         writer.sheets[rateType+"-"+freq].conditional_format('B2:B11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('C2:C11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('D2:D11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('E2:E11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('F2:F11', {'type': '3_color_scale'})


        # except:
        #     with pd.ExcelWriter("/Users/matthewmassicotte/Documents/GitHub/RatesBeta/FullSampleStats.xlsx") as writer:
        #         aggregate_fs[aggregate_fs.index>0].to_excel(writer, sheet_name=rateType+"-"+freq, index=True)
        #         writer.sheets[rateType+"-"+freq].set_column('A:J', 15)
        #         writer.sheets[rateType+"-"+freq].conditional_format('B2:B11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('C2:C11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('D2:D11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('E2:E11', {'type': '3_color_scale'})
        #         writer.sheets[rateType+"-"+freq].conditional_format('F2:F11', {'type': '3_color_scale'})
                
        # writer.close()


# # import os
# print(os.path.abspath(__file__))
    
            