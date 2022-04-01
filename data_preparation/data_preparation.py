import pandas as pd
import os
import unidecode
import re

path_to_data = 'data/'

#Import DBS motor outcome file
def Motor_outcomes(file_name):
    df = pd.read_excel(file_name, skiprows=[0])
    df = df.dropna()
    return df

m_o = Motor_outcomes('PL_DBS_outcomes.xlsx')

#Anonymizing patient data
def Anonymize(m_o):
    #Changing file names in the data directory
    i_names = []
    filenames = []
    data = []
    m_o.to_csv("Number_Name.csv", columns = ['Pavardė, vardas'])
    for index, row in m_o.iterrows():
        i_names.append((index, unidecode.unidecode(row[0].split()[0])))
    for filename in os.listdir(path_to_data):
        data.append(filename.split("-"))
        filenames.append(filename)
    for i in range(len(i_names)):
        for j in range(len(data)):
            if data[j][0].lower() in i_names[i][1].lower():
                data[j][0] = "Patient%i" % (i_names[i][0])
    for i in range(len(filenames)):
        old_name = path_to_data + filenames[i]
        new_name = path_to_data + '_'.join([part for part in data[i]])
        os.rename(old_name, new_name)
    m_o=m_o.drop(columns=['Pavardė, vardas'])
    m_o.to_csv("Motor_outcomes.csv")

#Combining 
def Combine(m_o):
    m_o['counts'] = 0
    m_o=m_o.drop(columns=['Pavardė, vardas'])
    f_df = pd.DataFrame()

    for i, filename in enumerate(os.listdir(path_to_data)):
        line_df = pd.DataFrame()
        patient = int(re.search(r'\d+', filename).group())
        m_o.loc[patient, 'counts'] += 1
        try:
            dfd = pd.read_csv(path_to_data + filename)
            for col in dfd.columns:
                if col[0:2] == 'lh':
                    names = ['lh' + col.split('_label')[1] + '_' + str(n) for n in range(37, len(dfd[col][:].tolist()))]
                    values = [dfd[col][37:].tolist()]
                    dft = pd.DataFrame(values, columns = names, index = [patient])
                    line_df = pd.concat([line_df, dft], axis = 1)
                if col[0:2] == 'rh':
                    names = ['rh' + col.split('_label')[1] + '_' + str(n) for n in range(37, len(dfd[col][:].tolist()))]
                    values = [dfd[col][37:].tolist()]
                    dft = pd.DataFrame(values, columns = names, index = [patient])
                    line_df = pd.concat([line_df, dft], axis = 1)
                if col[0:2] == 'Th':
                    names = ['Th' + col.split('_label')[1] + '_' + str(n) for n in range(37, len(dfd[col][:].tolist()))]
                    values = [dfd[col][37:].tolist()]
                    dft = pd.DataFrame(values, columns = names, index = [patient])
                    line_df = pd.concat([line_df, dft], axis = 1)
        except:
            print(filename)
        line_df['Patient'] = patient
        f_df = pd.concat([f_df, line_df], axis=0)
    f_df = f_df.groupby(['Patient']).sum(min_count=1)
    f_df.to_csv('Combined_radio.csv')
    return f_df

def final_combo(rad, m_o, psi):
    final = m_o.merge(rad, left_on=m_o.index, right_on='Patient', right_index=True)
    # print(final)
    final = final.merge(psi, on='Kodas')
    final.set_index('key_0', inplace=True, drop=True)
    final.index.names = ['Patient']
    # print(final)
    final.to_csv('Final_Data_check.csv')
    final = final.drop(columns=['Pavardė, vardas', 'Kodas'])
    final.index.names = ['']
    cols_at_end = ['Efektas DBS (1-blogas, 2-geras, 3-labai geras)']
    final = final[[c for c in final if c not in cols_at_end] + cols_at_end]
    final = final[[c for c in final if c not in cols_at_end] 
            + [c for c in cols_at_end if c in final]]
    final.to_csv('Final_Data_index.csv')


# Anonymize(m_o)
# rad = Combine(m_o)
rad = pd.read_csv('Combined_radio.csv', index_col=('Patient'))
print(rad)
psi = pd.read_csv('DBS_straipsniui_sulietas_darbinis.csv')
print(psi)
print(m_o)
final_combo(rad, m_o, psi)
