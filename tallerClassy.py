# integrantes: GABRIEL GOMEZ, EDUARDO DE LA HOZ, STEPHANIA DE LA HOZ

import numpy as np
import pandas as pd

a, b = 'áéíóúüñÁÉÍÓÚÜÑ', 'aeiouunAEIOUUN'
trans = str.maketrans(a, b)
# Read in tweet data
tweetsread = 'tuits_bayes.txt'
tweets = pd.read_csv(tweetsread, header=0, encoding='latin_1')

tweets["text"] = tweets["text"].str.replace('http\S+/\S+', '')

tweets["text"] = tweets["text"].str.replace(r'\b(\w{1,2})\b', '')

tweets["text"] = tweets["text"].str.translate(trans)

tweets["text"] = tweets["text"].str.lower()

tweets["text"] = tweets["text"].str.replace('[^a-z A-Z]', '')

tweets = tweets.groupby("screen_name").filter(lambda grupo: len(grupo) >= 5)

p_train = 0.80  # Porcentaje de train.

tweets['entrenamiento'] = np.random.uniform(0, 1, len(tweets)) <= p_train
train, test = tweets[tweets['entrenamiento'] ==
                     True], tweets[tweets['entrenamiento'] == False]
tweets = tweets.drop('entrenamiento', 1)

print("Datos usados para el entrenamiento: ", len(train))
print("Datos usados para la prueba: ", len(test))

print("\n")

print("por favor espere...")
print("esto puede tardar varios segundos o minutos")

repe = train["screen_name"].value_counts()
proApior = (repe[0:len(repe)]/len(train))*100

aux = train.copy()
desgrupado = aux.drop(labels='entrenamiento', axis=1)
dfgTrain = desgrupado.groupby(["screen_name"])
alfonzo = dfgTrain.get_group("lopezobrador_")
unam = dfgTrain.get_group("UNAM_MX")
microsoft = dfgTrain.get_group("MSFTMexico")
cmll = dfgTrain.get_group("CMLL_OFICIAL")

counts = dict()
for groupby in train.groupby("screen_name"):
    for line in alfonzo["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFAlfon = pd.DataFrame(counts, index=[0]).T

counts = dict()
for groupby in train.groupby("screen_name"):
    for line in unam["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFunam = pd.DataFrame(counts, index=[0]).T

counts = dict()
for groupby in train.groupby("screen_name"):
    for line in microsoft["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFmicrosoft = pd.DataFrame(counts, index=[0]).T

counts = dict()
for groupby in train.groupby("screen_name"):
    for line in cmll["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFcmll = pd.DataFrame(counts, index=[0]).T

# alfounam
alfoDF = nuevoDFAlfon.copy()
trigger = True
for palaUnam in nuevoDFunam.index:
    for palabraAlfo in nuevoDFAlfon.index:
        if palaUnam == palabraAlfo:
            trigger = False
    if trigger:
        alfoDF.append(nuevoDFunam.loc[[palaUnam]])
        alfoDF.at[palaUnam, 0] = 0
        alfoDF = alfoDF.add(1)
        trigger = True

# alfomicro
trigger = True
for palaMicro in nuevoDFmicrosoft.index:
    for palabraAlfo in nuevoDFAlfon.index:
        if palaMicro == palabraAlfo:
            trigger = False
    if trigger:
        alfoDF.append(nuevoDFmicrosoft.loc[[palaMicro]])
        alfoDF.at[palaMicro, 0] = 0
        alfoDF = alfoDF.add(1)
        trigger = True

# alfocmll
trigger = True
for palaCmll in nuevoDFcmll.index:
    for palabraAlfo in nuevoDFAlfon.index:
        if palaCmll == palabraAlfo:
            trigger = False
    if trigger:
        alfoDF.append(nuevoDFcmll.loc[[palaCmll]])
        alfoDF.at[palaCmll, 0] = 0
        alfoDF = alfoDF.add(1)
        trigger = True

# unamAmlo
unamDF = nuevoDFunam.copy()
trigger = True
for palabraAlfo in nuevoDFAlfon.index:
    for palaUnam in nuevoDFunam.index:
        if palaUnam == palabraAlfo:
            trigger = False
    if trigger:
        unamDF.append(nuevoDFAlfon.loc[[palabraAlfo]])
        unamDF.at[palabraAlfo, 0] = 0
        unamDF = unamDF.add(1)
        trigger = True

# unamMicro
trigger = True
for palabraMicro in nuevoDFmicrosoft.index:
    for palaUnam in nuevoDFunam.index:
        if palaUnam == palabraMicro:
            trigger = False
    if trigger:
        unamDF.append(nuevoDFmicrosoft.loc[[palabraMicro]])
        unamDF.at[palabraMicro, 0] = 0
        unamDF = unamDF.add(1)
        trigger = True

# unamcmll
trigger = True
for palabraCmll in nuevoDFcmll.index:
    for palaUnam in nuevoDFunam.index:
        if palaUnam == palabraCmll:
            trigger = False
    if trigger:
        unamDF.append(nuevoDFcmll.loc[[palabraCmll]])
        unamDF.at[palabraCmll, 0] = 0
        unamDF = unamDF.add(1)
        trigger = True

# microAmlo
microDF = nuevoDFmicrosoft.copy()
trigger = True
for palabraAlfo in nuevoDFAlfon.index:
    for palabraMicro in nuevoDFmicrosoft.index:
        if palabraMicro == palabraAlfo:
            trigger = False
    if trigger:
        microDF.append(nuevoDFAlfon.loc[[palabraAlfo]])
        microDF.at[palabraAlfo, 0] = 0
        microDF = microDF.add(1)
        trigger = True

# microUnam
trigger = True
for palaUnam in nuevoDFunam.index:
    for palabraMicro in nuevoDFmicrosoft.index:
        if palabraMicro == palaUnam:
            trigger = False
    if trigger:
        microDF.append(nuevoDFunam.loc[[palaUnam]])
        microDF.at[palaUnam, 0] = 0
        microDF = microDF.add(1)
        trigger = True

# microCmll
trigger = True
for palabraCmll in nuevoDFcmll.index:
    for palabraMicro in nuevoDFmicrosoft.index:
        if palabraMicro == palabraCmll:
            trigger = False
    if trigger:
        microDF.append(nuevoDFcmll.loc[[palabraCmll]])
        microDF.at[palabraCmll, 0] = 0
        microDF = microDF.add(1)
        trigger = True

# cmllAmlo
cmllDF = nuevoDFcmll.copy()
trigger = True
for palabraAlfo in nuevoDFAlfon.index:
    for palabraCmll in nuevoDFcmll.index:
        if palabraCmll == palabraAlfo:
            trigger = False
    if trigger:
        cmllDF.append(nuevoDFAlfon.loc[[palabraAlfo]])
        cmllDF.at[palabraAlfo, 0] = 0
        cmllDF = cmllDF.add(1)
        trigger = True

# cmllMicro
trigger = True
for palabraMicro in nuevoDFmicrosoft.index:
    for palabraCmll in nuevoDFcmll.index:
        if palabraCmll == palabraMicro:
            trigger = False
    if trigger:
        cmllDF.append(nuevoDFmicrosoft.loc[[palabraMicro]])
        cmllDF.at[palabraMicro, 0] = 0
        cmllDF = cmllDF.add(1)
        trigger = True

# cmllUnam
trigger = True
for palaUnam in nuevoDFunam.index:
    for palabraCmll in nuevoDFcmll.index:
        if palabraCmll == palaUnam:
            trigger = False
    if trigger:
        cmllDF.append(nuevoDFunam.loc[[palaUnam]])
        cmllDF.at[palaUnam, 0] = 0
        cmllDF = cmllDF.add(1)
        trigger = True

auxTest = test.copy()
desaugrupaTest = auxTest.drop(labels='entrenamiento', axis=1)
dfgTest = desaugrupaTest.groupby(["screen_name"])
alfonzoTest = dfgTest.get_group("lopezobrador_")
unamTest = dfgTest.get_group("UNAM_MX")
microsoftTest = dfgTest.get_group("MSFTMexico")
cmllTest = dfgTest.get_group("CMLL_OFICIAL")

counts = dict()
for groupby in test.groupby("screen_name"):
    for line in alfonzoTest["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFAlfonTest = pd.DataFrame(counts, index=[0]).T

counts = dict()
for groupby in test.groupby("screen_name"):
    for line in unamTest["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFUnamTest = pd.DataFrame(counts, index=[0]).T

counts = dict()
for groupby in test.groupby("screen_name"):
    for line in microsoftTest["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFMicrosoftTest = pd.DataFrame(counts, index=[0]).T

counts = dict()
for groupby in test.groupby("screen_name"):
    for line in cmllTest["text"]:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1


nuevoDFCmllTest = pd.DataFrame(counts, index=[0]).T

# 20amloUnam
trigger = True
for palaUnam in nuevoDFUnamTest.index:
    for palabraAlfo in alfoDF.index:
        if palaUnam == palabraAlfo:
            trigger = False
    if trigger:
        alfoDF.append(nuevoDFUnamTest.loc[[palaUnam]])
        alfoDF.at[palaUnam, 0] = 0
        alfoDF = alfoDF.add(1)
        trigger = True

# 20amloMicro
trigger = True
for palaMicro in nuevoDFMicrosoftTest.index:
    for palabraAlfo in alfoDF.index:
        if palaMicro == palabraAlfo:
            trigger = False
    if trigger:
        alfoDF.append(nuevoDFMicrosoftTest.loc[[palaMicro]])
        alfoDF.at[palaMicro, 0] = 0
        alfoDF = alfoDF.add(1)
        trigger = True

# 20amloCmll
trigger = True
for palaCmll in nuevoDFCmllTest.index:
    for palabraAlfo in alfoDF.index:
        if palaCmll == palabraAlfo:
            trigger = False
    if trigger:
        alfoDF.append(nuevoDFCmllTest.loc[[palaCmll]])
        alfoDF.at[palaCmll, 0] = 0
        alfoDF = alfoDF.add(1)
        trigger = True

# 20unamAmlo
trigger = True
for palabraAlfo in nuevoDFAlfonTest.index:
    for palaUnam in unamDF.index:
        if palaUnam == palabraAlfo:
            trigger = False
    if trigger:
        unamDF.append(nuevoDFAlfonTest.loc[[palabraAlfo]])
        unamDF.at[palabraAlfo, 0] = 0
        unamDF = unamDF.add(1)
        trigger = True

# 20unamMicro
trigger = True
for palaMicro in nuevoDFMicrosoftTest.index:
    for palaUnam in unamDF.index:
        if palaUnam == palaMicro:
            trigger = False
    if trigger:
        unamDF.append(nuevoDFMicrosoftTest.loc[[palaMicro]])
        unamDF.at[palaMicro, 0] = 0
        unamDF = unamDF.add(1)
        trigger = True

# 20unamCmll
trigger = True
for palaCmll in nuevoDFCmllTest.index:
    for palaUnam in unamDF.index:
        if palaUnam == palaCmll:
            trigger = False
    if trigger:
        unamDF.append(nuevoDFCmllTest.loc[[palaCmll]])
        unamDF.at[palaCmll, 0] = 0
        unamDF = unamDF.add(1)
        trigger = True

# 20microAmlo
trigger = True
for palabraAlfo in nuevoDFAlfonTest.index:
    for palabraMicro in microDF.index:
        if palabraMicro == palabraAlfo:
            trigger = False
    if trigger:
        microDF.append(nuevoDFAlfonTest.loc[[palabraAlfo]])
        microDF.at[palabraAlfo, 0] = 0
        microDF = microDF.add(1)
        trigger = True

# 20microUnam
trigger = True
for palaUnam in nuevoDFUnamTest.index:
    for palabraMicro in microDF.index:
        if palabraMicro == palaUnam:
            trigger = False
    if trigger:
        microDF.append(nuevoDFUnamTest.loc[[palaUnam]])
        microDF.at[palaUnam, 0] = 0
        microDF = microDF.add(1)
        trigger = True

# 20microCmll
trigger = True
for palaCmll in nuevoDFCmllTest.index:
    for palabraMicro in microDF.index:
        if palabraMicro == palaCmll:
            trigger = False
    if trigger:
        microDF.append(nuevoDFCmllTest.loc[[palaCmll]])
        microDF.at[palaCmll, 0] = 0
        microDF = microDF.add(1)
        trigger = True

# 20cmllAmlo
trigger = True
for palabraAlfo in nuevoDFAlfonTest.index:
    for palabraCmll in cmllDF.index:
        if palabraCmll == palabraAlfo:
            trigger = False
    if trigger:
        cmllDF.append(nuevoDFAlfonTest.loc[[palabraAlfo]])
        cmllDF.at[palabraAlfo, 0] = 0
        cmllDF = cmllDF.add(1)
        trigger = True

# 20cmllUnam
trigger = True
for palaUnam in nuevoDFUnamTest.index:
    for palabraCmll in cmllDF.index:
        if palabraCmll == palaUnam:
            trigger = False
    if trigger:
        cmllDF.append(nuevoDFUnamTest.loc[[palaUnam]])
        cmllDF.at[palaUnam, 0] = 0
        cmllDF = cmllDF.add(1)
        trigger = True

# 20cmllMicro
trigger = True
for palabraMicro in nuevoDFMicrosoftTest.index:
    for palabraCmll in cmllDF.index:
        if palabraCmll == palabraMicro:
            trigger = False
    if trigger:
        cmllDF.append(nuevoDFMicrosoftTest.loc[[palabraMicro]])
        cmllDF.at[palabraMicro, 0] = 0
        cmllDF = cmllDF.add(1)
        trigger = True

proAmlo = (alfoDF[0:len(alfoDF)]/len(alfoDF))*100
proUnam = (unamDF[0:len(unamDF)]/len(unamDF))*100
proMicro = (microDF[0:len(microDF)]/len(microDF))*100
proCmll = (cmllDF[0:len(cmllDF)]/len(cmllDF))*100

dataTabla=pd.DataFrame(index=proApior.index,columns=test.index)

for indiceRandom in test.index:
    counts = dict()
    words = test.at[indiceRandom, 'text'].split()
    for word in words:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1

    tweetprueba = pd.DataFrame(counts, index=[0]).T

    protenciaAmlo = []
    for palabraTest in tweetprueba.index:
        for palabraPc in proAmlo.index:
            if palabraTest == palabraPc:
                rest = pow(proAmlo.at[palabraPc, 0],
                        tweetprueba.at[palabraTest, 0])
                protenciaAmlo.append(rest)
                break

    protenciaUnam = []
    for palabraTest in tweetprueba.index:
        for palabraPc in proUnam.index:
            if palabraTest == palabraPc:
                rest = pow(proUnam.at[palabraPc, 0],
                        tweetprueba.at[palabraTest, 0])
                protenciaUnam.append(rest)
                break

    protenciaMicro = []
    for palabraTest in tweetprueba.index:
        for palabraPc in proMicro.index:
            if palabraTest == palabraPc:
                rest = pow(proMicro.at[palabraPc, 0],
                        tweetprueba.at[palabraTest, 0])
                protenciaMicro.append(rest)
                break

    protenciaCmll = []
    for palabraTest in tweetprueba.index:
        for palabraPc in proCmll.index:
            if palabraTest == palabraPc:
                rest = pow(proCmll.at[palabraPc, 0],
                        tweetprueba.at[palabraTest, 0])
                protenciaCmll.append(rest)
                break

    producAmlo = np.prod(protenciaAmlo)
    producUnam = np.prod(protenciaUnam)
    producMicro = np.prod(protenciaMicro)
    producCmll = np.prod(protenciaCmll)

    probabilidadAmlo = producAmlo*proApior.at['lopezobrador_']
    probabilidadUnam = producUnam*proApior.at['UNAM_MX']
    probabilidadMicro = producMicro*proApior.at['MSFTMexico']
    probabilidadCmll = producCmll*proApior.at['CMLL_OFICIAL']

    dataTabla.at['lopezobrador_',indiceRandom]=probabilidadAmlo
    dataTabla.at['UNAM_MX',indiceRandom]=probabilidadUnam
    dataTabla.at['MSFTMexico',indiceRandom]=probabilidadMicro
    dataTabla.at['CMLL_OFICIAL',indiceRandom]=probabilidadCmll
print(dataTabla)
print('\n')
print(test)
with pd.ExcelWriter('testOutput.xlsx') as writer:
    dataTabla.to_excel(writer,sheet_name='probabilidades')
    test.to_excel(writer,sheet_name='tweets de prueba')
print('\n')
print('se exporta un excel con la tabla de probabilidades y la tabla de tweets de prueba')
