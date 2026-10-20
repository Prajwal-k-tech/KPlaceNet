"""Collect licensed, geotagged candidate metadata, not a training-ready dataset.

Coordinate correctness, near duplicates, scene suitability and spatial split
separation still require review. Does not download photos or alter training.
"""
import argparse,csv,html,json,re,time
from pathlib import Path
import requests

TOWNS={'Kasaragod':(12.50,74.99),'Kannur':(11.87,75.37),
       'Kozhikode':(11.26,75.78),'Thrissur':(10.53,76.21),
       'Kochi':(9.93,76.27),'Kottayam':(9.59,76.52),
       'Munnar':(10.09,77.06),'Thiruvananthapuram':(8.52,76.94)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-town',type=int,default=50)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers['User-Agent']='KPlaceNet-academic-dataset-review/1.0'
    rows={};responses={}
    for town,(lat,lon) in TOWNS.items():
        params=dict(action='query',format='json',generator='geosearch',ggsprimary='primary',
            ggsnamespace=6,ggsradius=10000,ggscoord=f'{lat}|{lon}',ggslimit=args.per_town,
            prop='coordinates|imageinfo',colimit='max',coprimary='primary',
            iiprop='url|extmetadata',iiurlwidth=640)
        cache=args.output/(town.lower()+'-api.json')
        if cache.exists():
            data=json.loads(cache.read_text())
        else:
            for attempt in range(3):
                response=session.get('https://commons.wikimedia.org/w/api.php',params=params,timeout=30)
                response.raise_for_status();data=response.json()
                if 'error' not in data:break
                time.sleep(2**(attempt+1))
            if 'error' in data:
                print(town,'unavailable',data['error']['code'],flush=True);continue
            cache.write_text(json.dumps(data,indent=2))
        responses[town]=data
        for page in data.get('query',{}).get('pages',{}).values():
            info=page.get('imageinfo',[{}])[0];meta=info.get('extmetadata',{})
            get=lambda key:html.unescape(re.sub('<[^>]+>','',str(meta.get(key,{}).get('value',''))))
            try:
                plat=float(get('GPSLatitude'));plon=float(get('GPSLongitude'))
            except ValueError:
                coords=page.get('coordinates',[])
                if not coords:continue
                plat=float(coords[0]['lat']);plon=float(coords[0]['lon'])
            license=get('LicenseShortName')
            if not (license.startswith('CC BY') or license=='CC0' or 'Public domain' in license):continue
            if not (8 <= plat <= 12.9 and 74.5 <= plon <=77.5):continue
            rows[str(page['pageid'])]=dict(page_id=page['pageid'],title=page['title'],
                searched_near=town,latitude=plat,longitude=plon,license=license,
                license_url=get('LicenseUrl'),author=get('Artist'),
                source_page=info.get('descriptionurl',''),image_url=info.get('url',''),
                thumbnail_url=info.get('thumburl',''),
                review_status='candidate_only_not_used_for_training_or_evaluation')
        print(town,'total_unique',len(rows),flush=True);time.sleep(.3)
    records=list(rows.values())
    if not records:raise RuntimeError('No candidate metadata retrieved; retry later')
    (args.output/'commons_candidate_metadata.json').write_text(json.dumps(records,indent=2))
    (args.output/'commons_api_responses.json').write_text(json.dumps(responses,indent=2))
    with (args.output/'commons_candidate_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('Candidate records:',len(records),'not training-ready',flush=True)

if __name__=='__main__':main()
