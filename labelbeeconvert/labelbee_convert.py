#!/usr/bin/env python3

import json
import pandas as pd
import argparse
import numpy as np
import re
import matplotlib.pyplot as plt

def load_tracks_json(filename):
    with open(filename,"r") as f:
        T=json.load(f)
    return T

def load_tags_json(filename):
    with open(filename,"r") as f:
        Tags=json.load(f)
    return Tags


def tracks_to_df(T):
    df = pd.DataFrame(columns=['frame','id','leaving','entering',
                               'pollen','walking','fanning',
                               'falsealarm','wrongid'])
    #df = df.astype("bool")
    df[['frame','id']] = df[['frame','id']].astype("int64")
    
    col_labels = ['leaving','entering','pollen','walking','fanning','falsealarm','wrongid']
    df[col_labels] = df[col_labels].astype('int')
    
    all_labels=[]
    all_ids=[]
    
    for frameDict in T:
        if (frameDict is None): continue
        for id in frameDict:
            item=frameDict[id]
            if ('labels' not in item):
                labelsstr=''
            else:
                labelsstr=item['labels']
            if (labelsstr == ''):
                labels=[]
            else:
                labels=labelsstr.split(',')
            for l in labels:
                if (l not in all_labels): all_labels.append(l)
            if (id not in all_ids): all_ids.append(id)
            df=df.append(dict(frame=int(item['frame']),
                              id=int(item['ID']),
                              labels=labelsstr,
                              leaving='leaving' in labels,
                              entering='entering' in labels,
                              pollen='pollen' in labels,
                              walking='walking' in labels,
                              fanning='fanning' in labels,
                              falsealarm='falsealarm' in labels,
                              wrongid='wrongid' in labels
                              )
                        ,ignore_index=True)   
    #print(all_labels)
    #print(all_ids)
    return df
    
def tags_to_df(Tags):
    df = pd.DataFrame(columns=['frame','id','hamming','c','p','g','dm'])
    df[['frame','id']]=df[['frame','id']].astype(np.int32)
    
    all_ids=[]
    
    frames = sorted(Tags.keys(), key=lambda x: int(x))
    LD = []
    
    #print('Loading to list of record...')
    for framestr in frames:
        frame=int(framestr)
        frameRecord = Tags[framestr]
        if (frameRecord is None): continue
        L = frameRecord['tags']
        for item in L:
            #item=L[i]
            if (item['id'] not in all_ids): all_ids.append(item['id'])
            D=dict(frame=frame, id=item['id'])
            D['hamming'] = item.get('hamming')
            D['c'] = item.get('c')
            D['p'] = item.get('p')
            D['g'] = item.get('g')
            D['dm'] = item.get('dm')
            LD.append(D)

    #df=df.append(D,ignore_index=True)   
    #print('Converting to DataFrame...')
    df=pd.DataFrame(LD,columns=['frame','id','hamming','c','p','g','dm'])   
    #print(all_labels)
    #print(all_ids)
    return df

def load_tracks_df(filename):
    with open(filename,"r") as f:
        T=json.load(f)
    return tracks_to_df(T)

def load_tags_df(filename):
    with open(filename,"r") as f:
        T=json.load(f)
    return tags_to_df(T)

def timestamping(df,timestring,fps=20):
    if (df.shape[0]==0): return df   # Abort if empty

    cols=['datetime','date','time']
    df = df.reindex(columns = np.append( df.columns.values, cols))    
    if (timestring is not None):
        t0=pd.Timestamp(timestring)
        df[['datetime']] = df[['datetime']].astype(pd.Timestamp)
        df[['datetime']]=t0+pd.to_timedelta(df['frame']/fps,unit='s')
        
        df[['date']]=df[['datetime']].applymap(lambda d: d.date())
        df[['time']]=df[['datetime']].applymap(lambda d: d.time())
    else:
        t0=pd.NaT # Not a Time = undefined
        df[cols] = df[cols].astype(pd.Timestamp)
        df[cols] = pd.NaT # Not a Time == undefined
    
    return df

def parse_filename(filename):
    result = re.match(r'.*?(C(\d\d)_((\d\d\d\d\d\d)(\d\d\d\d\d\d)))', filename)
    if (result is None): return None # Default
    videoname=result.group(1)
    camera_id=result.group(2)
    timestamp='20'+result.group(3)
    date=result.group(4)
    time=result.group(5)
    return {'videoname': videoname, 
            'camera_id': camera_id,
            'timestamp': timestamp,
            'date': date,
            'time': time}

def timestamp_from_filename(filename):
    info = parse_filename(filename)
    if (info is None): return None
    return info['timestamp']

def load_fileset(inputlist):
    '''
    ex: inputlist="/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python/inputlist.csv"
    '''
    L = pd.read_csv(inputlist,
                header=0,names=['filename'])
    L[['timestamp']]=L[['filename']].applymap(timestamp_from_filename)
    
    df_list=[]
    for index, fileinfo in L.iterrows():
        filename=fileinfo['filename']
        print("Loading {}...".format(filename))
        
        infos=parse_filename(filename)
        if (infos is None):
            print("WARNING: input file '{}' not in format *Cxx_yymmddHHMMSS*. Timestamps will not be computed.".format(filename))
        
        #T=load_tracks_json(filename)
        #df1=tracks_to_df(T)
        df1 = load_tracks_df(filename)
        df1=timestamping(df1, fileinfo['timestamp'])
        if (infos is None):
            df1['video']='UNKNOWN'
        else:
            df1['video']=infos['videoname']
        df_list.append(df1)
        
    print("Merging into single DataFrame...")
    df = pd.concat(df_list)

    #df[['datetime']]=df[['time']].applymap(lambda d: pd.to_datetime(d))
        
    #df = df.query('FA!=True')
    
    return df

def load_tags_fileset(inputlist):
    '''
    ex: inputlist="/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python/tags_files.csv"
    '''
    L = pd.read_csv(inputlist,
                header=0,names=['filename'])
    L[['timestamp']]=L[['filename']].applymap(timestamp_from_filename)
    
    df_list=[]
    for index, fileinfo in L.iterrows():
        filename=fileinfo['filename']
        print("Loading {}...".format(filename))
        
        infos=parse_filename(filename)
        
        #T=load_tracks_json(filename)
        #df1=tracks_to_df(T)
        df1 = load_tags_df(filename)
        df1=timestamping(df1, fileinfo['timestamp'])
        if (infos is None):
            df1['video']=''
        else:
            df1['video']=infos['videoname']
        if (df1.shape[0]>0): # Do not add empty DataFrames
            df_list.append(df1)
        else:
            print('Warning: empty data from "{}"'.format(filename))
        
    print("Merging into single DataFrame...")
    df = pd.concat(df_list)
    
    #df.index = range(df.shape[0])

    df[['datetime']]=df[['time']].applymap(lambda d: pd.to_datetime(d))
        
    #df = df.query('FA!=True')
    
    return df


def plot_activities(df):
    
    import matplotlib.dates as mdates

    ids=df['id'].unique()
    ids.sort()
    
    rmap = {id: i for i,id in enumerate(ids)}    
    df[['uid']]=df[['id']].applymap(lambda x: rmap[x])
    
    df['FA']=df['falsealarm'] | df['wrongid']
    
    fig=plt.figure()
    idx=df.query('not falsealarm and not wrongid and not walking and not fanning and not pollen and not entering and not leaving').index
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'.',c='k',label='other')
    idx=df.query('falsealarm or wrongid').index
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'x',c='#a0a0a0',label='FA',mfc='none')
    idx=df.index[df['walking'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'o',c='k',label='walking',mfc='none')
    idx=df.index[df['fanning'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'s',c='brown',label='fanning',mfc='none')
    idx=df.index[df['pollen'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'s',c='#EEC000',label='pollen',linewidth=3)
    idx=df.index[df['entering'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'<',c='r',label='entering',mfc='none')
    idx=df.index[df['leaving'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'>',c='b',label='leaving',mfc='none')
    ax=plt.gca()
    ax.set_xlim(df.iloc[0]['datetime'],df.iloc[-1]['datetime'])
    plt.xticks(rotation='vertical')
    
    days = mdates.DayLocator()  
    hours = mdates.HourLocator()  
    dayFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    hourFmt = mdates.DateFormatter('%H:%M')
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dayFmt)
    #ax.xaxis.set_minor_locator(hours)
    #ax.xaxis.set_minor_formatter(hourFmt)

    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids)
    ax.grid(color='#888888', linestyle='-', linewidth=1)
    ax.legend()
    
    return fig

def main(args):
    evts=load_fileset(args.inputlist)
    
    #flagcols = ['leaving', 'entering', 'pollen', 'walking', 'fanning', 'falsealarm', 'wrongid']
    #evts[flagcols] = evts[flagcols].astype(int)
    
    if (args.plot):
        evts_val = evts.query('falsealarm!=True & wrongid!=True')
        evts_val.index = range(evts_val.shape[0])
        evts_val.reindex()
        fig=plot_activities(evts_val);
        plt.tight_layout();
        plt.show()

    if (args.output):
        evts.to_csv(args.output)

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    
    show_default = ' (default %(default)s)'
    parser.add_argument('-il',dest="inputlist",required=True,help="Input list as CSV")
    parser.add_argument('-o',dest="output",help="Output file")
    parser.add_argument('-plot', dest='plot', default=False, 
                        action='store_true',
                        help='Plot the merged events '+ show_default)

    args = parser.parse_args()
	
    main(args)

