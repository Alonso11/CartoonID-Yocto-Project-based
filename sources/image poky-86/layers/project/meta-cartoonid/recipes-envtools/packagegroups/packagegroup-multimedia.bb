SUMMARY = "Multimedia and development packages"
LICENSE = "MIT"

inherit packagegroup

# Use runtime package names (the names after "->" in the error messages)
RDEPENDS:${PN} = " \
    python3 \
    python3-pip \
    python3-dev \
    python3-modules \
    python3-opencv \
    python3-numpy \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    opencv\
    micromamba \
    bash \
    file \
    ldd \
    xserver-xorg \
    xinit \
    xauth \
    xinput \
    xkeyboard-config \
    git \
    wget \
    liberation-fonts \
    mesa\
"

# Add architecture-specific packages with their runtime names
RDEPENDS:${PN}:append:x86 = " \
    libc6 \
    libc6-utils \
    libfontconfig1 \
    libdrm2 \
    libgcc1 \
    libglu1 \
    libstdc++6 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
"

RDEPENDS:${PN}:append:x86-64 = " \
    libc6 \
    libc6-utils \
    libfontconfig1 \
    libdrm2 \
    libgcc1 \
    libglu1 \
    libstdc++6 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    xserver-xorg-driver-vesa \
    xserver-xorg-driver-modesetting \
    xserver-xorg-extension-glx \
    xserver-xorg-extension-dri \
    xserver-xorg-extension-extmod \
"



DESCRIPTION = "Package group for multimedia applications and development tools"

# This helps avoid some of the packaging issues
ALLOW_EMPTY:${PN} = "1"
