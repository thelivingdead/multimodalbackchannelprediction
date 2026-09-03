# Cropper audit

The withdrawn 3 s VideoMAE crops were produced by
`scripts/fetch_rgb_windows_nod3s.py` calling
`scripts/fetch_rgb_windows.py:crop_window`.

## How the old cropper chose a face

`crop_window` takes the middle frame of the 16-frame sample, runs OpenCV Haar
`haarcascade_frontalface_default`, and if any faces are found it keeps

    max(faces, key=lambda b: int(b[2]) * int(b[3]))

That is the largest box by area. The gold `person` field is written into the
npz after the fact and is never read when the box is chosen. `watch_list.csv`
is not read at all in this function.

If no face is found, the cropper falls back to a centred square of side
`min(width, height)`. On a 1280-wide RealTalk frame that square covers both
people.

The same function is the 60 s cropper. The 3 s fetcher imports it.

## Why that chose the wrong person in 51 percent of windows

RealTalk clips are two-person conversations. Haar often sees both faces. The
speaker is frequently closer to the camera, so their box is larger, and the
annotator was told to ignore that person. Largest-face selection therefore
locks onto the excluded speaker. The later wrong-half audit measured this at
51 percent of 3 s DEV windows, and at 6 of 15 TEST clips in the 60 s crops.

This file does not change those old result directories.
