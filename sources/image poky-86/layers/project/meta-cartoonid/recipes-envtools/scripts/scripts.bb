SUMMARY = "Colección de scripts útiles"
DESCRIPTION = "Scripts personalizados para el proyecto CartoonID"
LICENSE = "CLOSED"

SRC_URI = " \
    file://script1.sh \
    file://set_gstreamer_envs.sh \
    file://plugins/ \
    file://best_openvino_model/ \
    file://best.pt \
    file://yolov8n.pt \
    file://result_frame_000689.jpg \
    file://frame_000689.jpg \
    file://xorg.conf \
"

RDEPENDS:${PN} = " \
    bash \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    xserver-xorg \
    xinit \
"

do_install() {
    # Install scripts to /usr/bin
    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/script1.sh ${D}${bindir}/script1.sh
    install -m 0755 ${WORKDIR}/set_gstreamer_envs.sh ${D}${bindir}/set_gstreamer_envs.sh



    # Install model files
    install -d ${D}/home/root/.config/general
    install -m 0644 ${WORKDIR}/best.pt ${D}/home/root/.config/general/
    install -m 0644 ${WORKDIR}/yolov8n.pt ${D}/home/root/.config/general/

    # Install image files
    install -d ${D}/home/root/.config/general/images
    install -m 0644 ${WORKDIR}/result_frame_000689.jpg ${D}/home/root/.config/general/images/
    install -m 0644 ${WORKDIR}/frame_000689.jpg ${D}/home/root/.config/general/images/
    
    install -d ${D}${sysconfdir}/X11
    install -m 0644 ${WORKDIR}/xorg.conf ${D}${sysconfdir}/X11/
    # Set ownership
    chown -R root:root ${D}/home/root

   # Copiar best_openvino_model si existe
    if [ -d ${WORKDIR}/best_openvino_model ]; then
        install -d ${D}/home/root/.config/openvino
        cp -r ${WORKDIR}/best_openvino_model/* ${D}/home/root/.config/openvino/
    fi

    # Copiar plugins si existe
    if [ -d ${WORKDIR}/plugins ]; then
        install -d ${D}/home/root/.config/plugins
        cp -r ${WORKDIR}/plugins/* ${D}/home/root/.config/plugins/
    fi
}

# Comprehensive file inclusion
FILES:${PN} += " \
    ${sysconfdir}/X11/xorg.conf \
    ${bindir}/script1.sh \
    ${bindir}/set_gstreamer_envs.sh \
    /home/root/.config/openvino/* \
    /home/root/.config/openvino/*/* \
    /home/root/.config/plugins/* \
    /home/root/.config/plugins/*/* \
    /home/root/.config/general/* \
    /home/root/.config/general/*/* \
    /home/root/.config/general/images/* \
    /home/root/.config/general/images/*/* \
"