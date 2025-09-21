SUMMARY = "Colección de scripts útiles"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=3b83ef96387f14655fc854ddc3c6bd57"

SRC_URI = " \
    file://script1.sh \
    file://script2.sh \
    file://config-file.conf \
    file://LICENSE \
"

S = "${WORKDIR}"

do_install() {
    # Instalar scripts
    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/script1.sh ${D}${bindir}/
    install -m 0755 ${WORKDIR}/script2.sh ${D}${bindir}/

    # Instalar archivo de configuración
    install -d ${D}${sysconfdir}/my-scripts
    install -m 0644 ${WORKDIR}/config-file.conf ${D}${sysconfdir}/my-scripts/
    
    # Crear directorio de datos
    install -d ${D}${localstatedir}/lib/mi-app
}
