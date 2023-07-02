import os

from gwpy.table import EventTable
from gwpy.time import tconvert


os.environ['GRAVITYSPY_DATABASE_USER'] = 'mla'
os.environ['GRAVITYSPY_DATABASE_PASSWD'] = 'gl1tch35Rb4d!'

st1, et1 = int(tconvert('2019-05-01 0:00:00')), int(tconvert('2019-06-01 0:00:00'))
et1a = int(tconvert('2019-08-01 0:00:00'))
st2, et2 = int(tconvert('2020-02-01 0:00:00')), int(tconvert('2020-03-01 0:00:00'))

ml_types = ['Tomte','Blip','Koi_Fish','Extremely_Loud','Blip_Low_Frequency']

gspy1 = EventTable.fetch('gravityspy', 'glitches_v2d0',
                            selection=[f"{st1}<event_time<{et1a} and ml_confidence > 0.9"],
                            host='gravityspyplus.ciera.northwestern.edu')

sel = [x['ml_label'] in ml_types for x in gspy1]
gspy1a = gspy1[sel]

gspy1a.write('gspy-3month.json', format='pandas.json')
