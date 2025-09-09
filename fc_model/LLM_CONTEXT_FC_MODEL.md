## LLM Context for the fc_model library (≥ 1.1.4)

Briefly: fc_model is a strictly typed object model for the Fidesys Case (.fc) format. Use only the public re-exports from the `fc_model` package. Do not import from internal modules `fc_model.fc_*`.

### Versions and compatibility
- Minimum version for proper typing and mypy support: 1.1.4.
- Python: ≥ 3.8.
- The distribution includes `py.typed` (PEP 561), so first-party types are available to static analysis tools.

### Import and usage rules
- Import only from the package root:
  - Example: `from fc_model import FCModel, FCMaterial, FCData`
  - Avoid: `from fc_model.fc_materials import FCMaterial`
- Serialization/deserialization:
  - `FCModel(filepath)` — load a .fc (JSON) file and decode binary fields (Base64).
  - `FCModel.dump()` — get a dictionary representation of the model.
  - `FCModel.save(path)` — write to a `.fc` file.

### Quick start
```python
from fc_model import FCModel

# Load a model from file
m = FCModel("path/to/model.fc")

# ... operate on model entities ...

# Save
m.save("path/to/output.fc")
```

### Public API map (essentials)
- Classes: `FCModel`, `FCMesh`, `FCBlock`, `FCCoordinateSystem`, `FCConstraint`,
  `FCMaterial`, `FCPropertyTable`, `FCLoad`, `FCRestraint`, `FCInitialSet`,
  `FCReceiver`, `FCSet`, `FCValue`, `FCData`, `FCDependencyColumn`.
- Constants/lookups (for codes and names):
  - Materials: `FC_MATERIAL_PROPERTY_NAMES_KEYS`, `FC_MATERIAL_PROPERTY_NAMES_CODES`,
    `FC_MATERIAL_PROPERTY_TYPES_KEYS`, `FC_MATERIAL_PROPERTY_TYPES_CODES`.
  - Loads/BCs/ICs: `FC_LOADS_TYPES_KEYS`, `FC_LOADS_TYPES_CODES`,
    `FC_RESTRAINT_FLAGS_KEYS`, `FC_RESTRAINT_FLAGS_CODES`,
    `FC_INITIAL_SET_TYPES_KEYS`, `FC_INITIAL_SET_TYPES_CODES`.
  - Mesh: `FC_ELEMENT_TYPES_KEYID`, `FC_ELEMENT_TYPES_KEYNAME`.
  - Dependencies: `FC_DEPENDENCY_TYPES_KEYS`, `FC_DEPENDENCY_TYPES_CODES`.

### Common recipes
- Load/save:
```python
from fc_model import FCModel
m = FCModel("case.fc")
m.save("out.fc")
```

- Access entities:
```python
# Root-level collections
_ = m.mesh, m.blocks, m.materials, m.loads, m.restraints
_ = m.initial_sets, m.nodesets, m.sidesets, m.receivers, m.settings
```

- Add a material property:
```python
from fc_model import FCMaterialProperty, FCData

mat_id = next(iter(m.materials))
mat = m.materials[mat_id]
prop = FCMaterialProperty(type="USUAL", name="DENSITY", data=FCData(data="", dep_type=0, dep_data=""))
mat.properties.setdefault("common", [[]])[0].append(prop)
```

- Work with sets:
```python
# Node and side sets
nodesets = m.nodesets
sidesets = m.sidesets
```

- Dump without writing to a file:
```python
data = m.dump()  # serializable dict structure
```

### Invariants and recommendations
- Import from the `fc_model` root and rely on type annotations.
- Use lookups/constants from `fc_model`; do not hardcode codes.
- `settings` is an object (dict), defaults to an empty dict.
- For serialization use only `dump()`/`save()`.

### Where to look for implementation details
- Entry point (re-export): `src/fc_model/__init__.py` — list of public classes and constants.
- Domain modules: `fc_mesh.py`, `fc_materials.py`, `fc_blocks.py`, `fc_data.py`,
  `fc_conditions.py`, `fc_constraint.py`, `fc_property_tables.py`, `fc_set.py`, `fc_receivers.py`.
- Usage example and round-trip: `tests/test_round_trip.py`.
- Published on PyPI: [fc_model — PyPI](https://pypi.org/project/fc-model/).

### Tips for RAG/embeddings
- Index first: `src/fc_model/__init__.py` (public surface), then `fc_mesh.py`, `fc_materials.py`,
  `fc_data.py`, `fc_blocks.py`.
- Chunk by entity: separate classes/methods/constants, preserve signatures and docstrings.
- Key search terms: "materials", "mesh", "loads", "restraints", "sets", and constant names from the public API.

### Ready-to-use "Context" block for LLM prompts
```text
Context:
- Library: fc_model (>=1.1.4). Purpose: strictly typed model of Fidesys Case (.fc).
- Use only public re-exports from `fc_model`. Do not import from internal modules.
- Key classes: FCModel, FCMesh, FCBlock, FCCoordinateSystem, FCConstraint, FCMaterial,
  FCPropertyTable, FCLoad, FCRestraint, FCInitialSet, FCReceiver, FCSet, FCValue, FCData, FCDependencyColumn.
- Constants: FC_MATERIAL_PROPERTY_*, FC_LOADS_TYPES_*, FC_RESTRAINT_FLAGS_*, FC_ELEMENT_TYPES_*, FC_DEPENDENCY_TYPES_*.
- Typical operations:
  - Load/save: m = FCModel("in.fc"); m.save("out.fc")
  - Materials: via m.materials[...] and FCMaterialProperty/FCData
  - Mesh/sets: m.mesh, m.nodesets, m.sidesets
- Invariants: rely on type annotations; dump()/save() are the only serialization methods;
  use constants from the public API.
Task:
[briefly describe what to do, which model sections, desired output]
Output:
1) Step-by-step plan
2) Code with imports from `fc_model`
3) Checks/invariants
```

### FAQ (useful for tools)
- Mypy says “Cannot find implementation or library stub for module 'fc_model'”:
  - Ensure `fc_model>=1.1.4` and clear cache: `mypy --clear-cache`.
  - External projects must not have a `fc_model.py` file/module overshadowing the package.

### Links
- PyPI: [fc_model — PyPI](https://pypi.org/project/fc-model/)
- Sources/structure: `src/fc_model/` directory in the repository.


