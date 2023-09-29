
names = [
'DUJUAN',
'SURIGAE',
'THREE_W',
'CHOI-WAN',
'KOGUMA',
'CHAMPI',
'SEVEN_W',
'EIGHT_W',
'IN-FA',
'CEMPAKA',
'NEPARTAK',
'TWELVE_W',
'LUPIT',
'MIRINAE',
'NIDA',
'OMAIS',
'SEVENTEEN_W',
'CONSON',
'CHANTHU',
'MINDULLE',
'DIANMU',
'LIONROCK',
'NAMTHEUN',
'KOMPASU',
'MALOU',
'TWENTYSIX_W',
'NYATOH',
'RAI',
'TWENTYNINE_W',
]

periods = [
'Feb 17,2021 - Feb 22,2021',
'Apr 13,2021 - Apr 25,2021',
'May 12,2021 - May 15,2021',
'May 29,2021 - Jun 06,2021',
'Jun 12,2021 - Jun 13,2021',
'Jun 21,2021 - Jun 28,2021',
'Jul 05,2021 - Jul 06,2021',
'Jul 07,2021 - Jul 08,2021',
'Jul 16,2021 - Jul 28,2021',
'Jul 18,2021 - Jul 24,2021',
'Jul 23,2021 - Jul 29,2021',
'Aug 02,2021 - Aug 06,2021',
'Aug 02,2021 - Aug 10,2021',
'Aug 04,2021 - Aug 10,2021',
'Aug 04,2021 - Aug 08,2021',
'Aug 10,2021 - Aug 24,2021',
'Sep 02,2021 - Sep 04,2021',
'Sep 06,2021 - Sep 13,2021',
'Sep 06,2021 - Sep 18,2021',
'Sep 22,2021 - Oct 02,2021',
'Sep 22,2021 - Sep 24,2021',
'Oct 07,2021 - Oct 11,2021',
'Oct 10,2021 - Oct 18,2021',
'Oct 08,2021 - Oct 15,2021',
'Oct 24,2021 - Oct 30,2021',
'Oct 26,2021 - Oct 27,2021',
'Nov 29,2021 - Dec 04,2021',
'Dec 13,2021 - Dec 21,2021',
'Dec 16,2021 - Dec 17,2021',
]

def str2period(inputs, is_end=False):
    month = {'Jan':'01',
             'Feb':'02',
             'May':'03',
             'Apr':'04',
             'May':'05',
             'Jun':'06',
             'Jul':'07',
             'Aug':'08',
             'Sep':'09',
             'Oct':'10',
             'Nov':'11',
             'Dec':'12'}
    inputs = inputs.replace(',', ' ').split(' ')
    if is_end:
        inputs[1] = int(inputs[1]) + 1
        inputs[1] = str(inputs[1])
    date = inputs[2] + month[inputs[0]] + inputs[1] + '0000'
    return date

f = open('typoon_2021.txt', 'w')
for i in range(len(names)):
    name = names[i]
    period = periods[i]
    start_, end_ = period.split(' - ')

    start = str2period(start_)
    end = str2period(end_, True)
    info = '{} {} {}'.format(name, start, end)
    print(info, file=f)
f.close()