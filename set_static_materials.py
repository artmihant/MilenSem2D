from fc_model import FCModel, FCMaterial, FCBlock, FCMaterialProperty, FCData, FCElement
import numpy as np

fc_model = FCModel('model2.fc')

ethalon_material_porps = fc_model.materials[1].properties

import csv

# Читаем данные из файла data/static_material.csv
static_material_props = []
with open('data/static_material.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        static_material_props.append(row)


for i in range(1,18):

    material_prop = static_material_props[i-1]

    fc_model.blocks[i].material_id = i

    material = FCMaterial({
        'id': i,
        'name': f'mat_{i}_{material_prop["name"].strip()}',
    })

    vp = float(material_prop['vp'].replace(',', '.'))
    vs = float(material_prop['vs'].replace(',', '.'))
    phob = float(material_prop['phob'].replace(',', '.'))*1000

    material.properties = {
        'elasticity': [[
            FCMaterialProperty(
                'HOOK', "VP",
                FCData(np.array([vp], np.float64), 0, '')
            ),
            FCMaterialProperty(
                'HOOK', "VS",
                FCData(np.array([vs], np.float64), 0, '')
            )
        ]],
        'common': [[
            FCMaterialProperty(
                'USUAL', "DENSITY",
                FCData(np.array([phob], np.float64), 0, '')
            )
        ]]
    }

    fc_model.materials[i] = material



# fc_model.save('model2.fc')