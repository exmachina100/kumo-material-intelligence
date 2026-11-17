# Private datasets — NOT committed

Expected structure:
```
training/data/
  train/metal/*.jpg
  train/polymer/*.jpg
  val/metal/*.jpg
  val/polymer/*.jpg
  test/metal/*.jpg
  test/polymer/*.jpg
```
Create splits by physical sample and source before assigning individual photographs, to prevent leakage from near-duplicate photos.

Do not add supplier confidential images or unlicensed datasets to GitHub. Never mix product shots from the same physical sample across splits. Confirm usage rights and consent.
