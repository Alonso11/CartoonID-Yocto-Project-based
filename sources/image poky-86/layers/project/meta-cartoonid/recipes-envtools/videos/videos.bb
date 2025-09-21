# recipes-multimedia/videos/videos.bb

SUMMARY = "Pre-downloaded video file"
DESCRIPTION = "Recipe to install a pre-downloaded video file"
LICENSE = "CLOSED"

# Set the version and source
PV = "1.0"
SRC_URI = "file://sherk1.mp4"

# Don't attempt to compile anything
do_compile() {
    :
}

# Install the video file
do_install() {
    install -d ${D}${datadir}/videos
    install -m 0644 ${WORKDIR}/sherk1.mp4 ${D}${datadir}/videos/
}

# Package the files
FILES:${PN} = "${datadir}/videos/sherk1.mp4"

# Prevent QA package errors
INSANE_SKIP:${PN} += "already-stripped"
