# RealWaste Intake Summary v1

## Purpose

This stage creates the intake plan and manifest builder for adding RealWaste as the next public dataset.

## Why This Is Needed

The current v1 pipeline is complete, but it is still based mainly on the TrashNet-style dataset.

RealWaste is added next to improve dataset diversity and prepare a stronger expanded-data model.

## Planned Mapping

| RealWaste Label     | OpenWaste-HR Role                     |
| ------------------- | ------------------------------------- |
| Cardboard           | paper_cardboard                       |
| Food Organics       | organic                               |
| Glass               | glass                                 |
| Metal               | metal                                 |
| Miscellaneous Trash | residual                              |
| Paper               | paper_cardboard                       |
| Plastic             | plastic                               |
| Textile Trash       | unknown-test / future class candidate |
| Vegetation          | organic                               |

## Important Safety Rule

Textile Trash is not forced into a current known class. It is treated as outside the current known taxonomy and kept for unknown/future-class use.

## Next Stage

After the RealWaste dataset is downloaded locally, the builder can create the RealWaste manifest. Then the next stage should inspect the manifest and prepare expanded training splits.
