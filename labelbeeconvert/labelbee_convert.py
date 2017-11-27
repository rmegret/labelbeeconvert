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
    df = pd.DataFrame(columns=['frame','id','leaving','entering','pollen','walking','fanning','FA'])
    df = df.astype("bool")
    df[['frame','id']] = df[['frame','id']].astype("int64")
    
    all_labels=[]
    all_ids=[]
    
    for frameDict in T:
        if (frameDict is None): continue
        for id in frameDict:
            item=frameDict[id]
            if (item['labels'] == ''):
                labels=[]
            else:
                labels=item['labels'].split(',')
            for l in labels:
                if (l not in all_labels): all_labels.append(l)
            if (id not in all_ids): all_ids.append(id)
            df=df.append(dict(frame=int(item['frame']),
                              id=int(item['ID']),
                              labels=item['labels'],
                              leaving='leaving' in labels,
                              entering='entering' in labels,
                              pollen='pollen' in labels,
                              walking='walking' in labels,
                              fanning='fanning' in labels,
                              FA='falsealarm' in labels or 'wronglabel' in labels
                              )
                        ,ignore_index=True)   
    #print(all_labels)
    #print(all_ids)
    return df
    
def tags_to_df(Tags):
    df = pd.DataFrame(columns=['frame','id','hamming','c','p'])
    df[['frame','id']]=df[['frame','id']].astype(np.int32)
    
    all_ids=[]
    
    for framestr in Tags:
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
            df=df.append(D,ignore_index=True)   
    #print(all_labels)
    #print(all_ids)
    return df

def timestamping(df,timestring,fps=20):
    t0=pd.Timestamp(timestring)
    df = df.reindex(columns = np.append( df.columns.values, ['time']))
    df[['time']] = df[['time']].astype(pd.Timestamp)
    df[['time']]=t0+pd.to_timedelta(df['frame']/fps,unit='s')
    
    df[['datetime']]=df[['time']].applymap(lambda d: pd.to_datetime(d))
    
    return df

def timestamp_from_filename(filename):
    result = re.match(r'.*?C(\d\d)_((\d\d\d\d\d\d)(\d\d\d\d\d\d))', filename)
    if (result is None): return None # Default
    videoname=result.group(0)
    camera_id=result.group(1)
    timestamp='20'+result.group(2)
    date=result.group(3)
    time=result.group(4)
    return timestamp

def load_fileset(inputlist):
    '''
    ex: inputlist="/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python/inputlist.csv"
    '''
    L = pd.read_csv(inputlist,
                header=0,names=['filename'])
    L[['timestamp']]=L[['filename']].applymap(timestamp_from_filename)
    
    df=pd.DataFrame()
    for index, row in L.iterrows():
        filename=row['filename']
        print("Loading {}...".format(filename))
        T=load_tracks_json(filename)
        df1=tracks_to_df(T)
        df1=timestamping(df1, row['timestamp'])
        df=df.append(df1,ignore_index=True)

    df[['datetime']]=df[['time']].applymap(lambda d: pd.to_datetime(d))
        
    #df = df.query('FA!=True')
    
    return df

def plot_activities(df):
    
    import matplotlib.dates as mdates

    ids=df['id'].unique()
    ids.sort()
    
    rmap = {id: i for i,id in enumerate(ids)}    
    df[['uid']]=df[['id']].applymap(lambda x: rmap[x])
    
    fig=plt.figure()
    idx=df.query('not FA and not walking and not fanning and not pollen and not entering and not leaving').index
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'.',c='k',label='other')
    idx=df.index[df['FA'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'x',c='#a0a0a0',label='FA/WId',mfc='none')
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
  file=pd.read_json(annotations,output_file)
  with open(output_file,"w") as f:
      json.dump(Annotations,f)

if __name__ == "__main__": 
	parser = argparse.ArgumentParser()
	parser.add_argument('-il',dest="inputlist",help="Input list as CSV")
	parser.add_argument('-o',dest="output",help="Output file")
	args = parser.parse_args()
	
	main(args)

