#!/usr/bin/env python3
"""
Скрипт для построения fc моделей с разными размерами сетки.

Поддерживает создание моделей с грубой сеткой путем усреднения свойств материала.
"""

import numpy as np
import sys
from pathlib import Path

# Добавляем путь к fc_model
sys.path.append(str(Path(__file__).parent / 'fc_model'))

from fc_model import FCModel, FCMaterial, FCBlock, FCMaterialProperty, FCData, FCElement


def load_original_data():
    """Загрузка оригинальных данных материалов и сетки."""
    print("Загрузка оригинальных данных...")

    # Загрузка материалов
    material_data = np.load('model1_material.npz')
    kriging_params = material_data['kriging_params']

    # Загрузка сетки
    mesh_data = np.load('model1_mesh_coords.npz', allow_pickle=True)
    x_coords = mesh_data['x_coordinates']
    z_coords = mesh_data['z_coordinates']

    print(f"Оригинальные материалы: {kriging_params.shape} (Vp, плотность, Vs)")
    print(f"Оригинальная сетка: {x_coords.shape[0]}×{z_coords.shape[1]} узлов")
    print(f"Оригинальные элементы: {(x_coords.shape[0]-1)}×{(z_coords.shape[1]-1)}")

    return kriging_params, x_coords, z_coords


def create_coarse_grid(dx_target, dz_target, kriging_params, x_coords, z_coords):
    """
    Создание грубой сетки с заданными размерами ячеек.

    Args:
        dx_target: желаемый размер ячейки по горизонтали (м)
        dz_target: желаемый размер ячейки по вертикали (м)
        kriging_params: массив свойств материала (nx, nz, 3)
        x_coords: координаты X узлов (nx,)
        z_coords: координаты Z узлов (nx, nz)

    Returns:
        tuple: (coarse_kriging, coarse_x, coarse_z)
    """
    print(f"Создание грубой сетки {dx_target}×{dz_target} м...")

    nx_orig, nz_orig = kriging_params.shape[0], kriging_params.shape[1]

    # Оригинальные размеры
    dx_orig = 10.0  # м (известно из предыдущих расчетов)
    dz_orig = 10.0  # м

    # Размеры грубой сетки
    nx_coarse = int(np.ceil((nx_orig * dx_orig) / dx_target))
    nz_coarse = int(np.ceil((nz_orig * dz_orig) / dz_target))

    print(f"Оригинальная сетка: {nx_orig}×{nz_orig} элементов")
    print(f"Грубая сетка: {nx_coarse}×{nz_coarse} элементов")

    # Создаем грубые массивы
    coarse_kriging = np.zeros((nx_coarse, nz_coarse, 3))
    coarse_x = np.linspace(x_coords[0], x_coords[-1], nx_coarse + 1)
    coarse_z = np.zeros((nx_coarse + 1, nz_coarse + 1))

    # Факторы агрегации
    x_factor = max(1, int(np.ceil(dx_target / dx_orig)))
    z_factor = max(1, int(np.ceil(dz_target / dz_orig)))

    print(f"Факторы агрегации: x={x_factor}, z={z_factor}")

    # Агрегация свойств материала
    for i in range(nx_coarse):
        for j in range(nz_coarse):
            # Границы блока в оригинальной сетке
            i_start = i * x_factor
            i_end = min((i + 1) * x_factor, nx_orig)
            j_start = j * z_factor
            j_end = min((j + 1) * z_factor, nz_orig)

            # Усреднение свойств в блоке
            block = kriging_params[i_start:i_end, j_start:j_end, :]
            coarse_kriging[i, j, :] = np.mean(block, axis=(0, 1))

            # Усреднение Z координат
            z_block = z_coords[i_start:i_end, j_start:j_end]
            if z_block.size > 0:
                coarse_z[i, j] = np.mean(z_block)

    # Заполняем границы
    coarse_z[:, -1] = coarse_z[:, -2] + dz_target
    coarse_z[-1, :] = coarse_z[-2, :]

    return coarse_kriging, coarse_x, coarse_z


def build_fc_model_from_arrays(x_coords, z_coords, vp, density, vs, output_path):
    """
    Построение fc модели из массивов данных.

    Args:
        x_coords: координаты X узлов
        z_coords: координаты Z узлов (nx, nz)
        vp: скорости Vp (nx-1, nz-1)
        density: плотности (nx-1, nz-1)
        vs: скорости Vs (nx-1, nz-1)
        output_path: путь для сохранения fc файла
    """
    print(f"Построение FC модели: {vp.shape[0]}×{vp.shape[1]} элементов")

    # Объединяем свойства в один массив
    kriging_params = np.stack([vp, density, vs], axis=2)

    # Создание модели
    model = FCModel()

    # Настройки для 2D упругой модели
    model.settings = {
        "type": "dynamic",
        "dimensions": "2D",
        "plane_state": "p-strain",
        "elasticity": True,
        "finite_deformations": False,
        "lumpmass": False,
        "damping": {
            "type": "rayleigh",
            "alpha": 0.0,
            "beta": 0.0
        }
    }

    # Добавление узлов
    nx, nz = x_coords.shape[0], z_coords.shape[1]

    # Векторизованное создание координат узлов
    x_flat = np.repeat(x_coords, nz)
    y_flat = z_coords.flatten()
    z_flat = np.zeros_like(x_flat)

    nodes_xyz = np.stack([x_flat, y_flat, z_flat], axis=1)
    node_ids = np.arange(1, nx * nz + 1, dtype=np.int32)

    model.mesh.nodes_ids = node_ids
    model.mesh.nodes_xyz = nodes_xyz

    # Добавление элементов
    element_id = 1
    elements_created = 0

    for i in range(nx - 1):
        for j in range(nz - 1):
            n1 = i * nz + j + 1
            n2 = (i + 1) * nz + j + 1
            n3 = (i + 1) * nz + (j + 1) + 1
            n4 = i * nz + (j + 1) + 1

            element = FCElement({
                'id': element_id,
                'type': 'QUAD4',
                'nodes': [n1, n2, n3, n4],
                'block': 1,
                'parent_id': 0,
                'order': 1
            })

            model.mesh.add(element)
            element_id += 1
            elements_created += 1

    # Создание материала
    num_elements = (nx - 1) * (nz - 1)

    # Векторные операции для расчета свойств
    vp_flat = vp.flatten()
    density_flat = density.flatten()
    vs_flat = vs.flatten()

    # Перевод плотности из г/см³ в кг/м³
    density_kgm3 = density_flat * 1000

    # Расчет модуля Юнга и коэффициента Пуассона
    vp2 = vp_flat ** 2
    vs2 = vs_flat ** 2

    young_moduli = density_kgm3 * vs2 * (3 * vp2 - 4 * vs2) / (vp2 - vs2)
    poisson_ratios = (vp2 - 2 * vs2) / (2 * (vp2 - vs2))
    densities = density_kgm3

    elements_ids = np.arange(1, num_elements + 1, dtype=np.float64)

    # Создаем материал с табличными свойствами
    material_dict = {
        'id': 1,
        'name': f'model_{nx-1}x{nz-1}_material',
    }

    material = FCMaterial(material_dict)
    material.properties = {
        'elasticity': [[
            FCMaterialProperty(
                'HOOK', "YOUNG_MODULE",
                FCData(young_moduli.astype(np.float64), [10], [elements_ids])
            ),
            FCMaterialProperty(
                'HOOK', "POISSON_RATIO",
                FCData(poisson_ratios.astype(np.float64), [10], [elements_ids])
            )
        ]],
        'common': [[
            FCMaterialProperty(
                'USUAL', "DENSITY",
                FCData(densities.astype(np.float64), [10], [elements_ids])
            )
        ]]
    }

    model.materials[1] = material

    # Создание блока
    block = FCBlock({
        'id': 1,
        'cs_id': 0,
        'material_id': 1,
        'property_id': 0
    })

    model.blocks[1] = block

    # Сохранение модели
    print(f"Сохранение модели в {output_path}...")
    model.save(output_path)

    return elements_created, len(node_ids)


def create_multiple_models():
    """Создание моделей с разными размерами сетки."""
    print("=" * 80)
    print("ПОСТРОЕНИЕ FC МОДЕЛЕЙ С РАЗНЫМИ РАЗМЕРАМИ СЕТКИ")
    print("=" * 80)

    # Загрузка оригинальных данных
    kriging_params, x_coords, z_coords = load_original_data()

    # Комбинации размеров сетки
    grid_sizes = [
        (10, 10),  # оригинальная
        (25, 10),
        (50, 10),
        (25, 20),
        (50, 20),
        (50, 40)
    ]

    results = []

    for dx, dz in grid_sizes:
        print(f"\n{'='*60}")
        print(f"СОЗДАНИЕ МОДЕЛИ {dx}×{dz} м")
        print(f"{'='*60}")

        try:
            # Создание грубой сетки
            coarse_kriging, coarse_x, coarse_z = create_coarse_grid(
                dx, dz, kriging_params, x_coords, z_coords
            )

            # Разделение свойств
            vp_coarse = coarse_kriging[:, :, 0]
            density_coarse = coarse_kriging[:, :, 1]
            vs_coarse = coarse_kriging[:, :, 2]

            # Создание имени файла
            output_file = f"model_{dx}x{dz}.fc"

            # Построение модели
            elements_count, nodes_count = build_fc_model_from_arrays(
                coarse_x, coarse_z, vp_coarse, density_coarse, vs_coarse, output_file
            )

            print(f"✓ Модель {dx}×{dz} создана успешно!")
            print(f"  Файл: {output_file}")
            print(f"  Элементов: {elements_count}")
            print(f"  Узлов: {nodes_count}")

            results.append({
                'dx': dx,
                'dz': dz,
                'file': output_file,
                'elements': elements_count,
                'nodes': nodes_count
            })

        except Exception as e:
            print(f"✗ ОШИБКА при создании модели {dx}×{dz}: {e}")
            import traceback
            traceback.print_exc()

    # Итоговый отчет
    print(f"\n{'='*80}")
    print("ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*80}")

    for result in results:
        print(f"{result['dx']:3d}×{result['dz']:2d} м: {result['elements']:6d} элементов, {result['nodes']:6d} узлов - {result['file']}")

    print(f"\nВсего создано моделей: {len(results)}/{len(grid_sizes)}")

    return results


def main():
    """Основная функция."""
    return create_multiple_models()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
