# 📚 Recursos y Referencias de la Investigación

Este documento recopila los recursos encontrados durante la investigación para el proyecto. 


## Como correr una imagen minima en virtualbox

Anadir en local.conf la variable:
	**IMAGE_FSTYPES += "wic.vid"**

### Resumen imagenes compatibles virtualbox

| Image Type | VirtualBox Compatibility  | Description                                       | Notes                                               |
|------------|---------------------------|-------------------------------------------------|-----------------------------------------------------|
| VMDK       | ✔️                | VirtualBox native disk format, often built by Yocto directly | Best for disk image use in VirtualBox                |
| VDI        | ✔️                 | VirtualBox native disk format                     | Also supported by Yocto, ideal for VirtualBox        |
| ISO (Live) | ✔️                      | Bootable live CD/DVD image                         | Used for live boot and testing, no persistence       |
| WIC        | ☑️       | Raw disk image for flashing hardware              | Must convert to VMDK or VDI for VirtualBox           |


### Path imagenes construidas
 **build/tmp/deploy/images/**

### Configure VirtualBox
Como configurar la maquina virtual:

**Virtual machine name and operating system**

- Agregar VM Name
- Seleccionar OS->Linux
  
**Specify virtual hardware**
- Seleccionar OS Version->Other Linux(64-bit)
- Seleccionar Base Memory->1024 MB
  
**Specify virtual hard disk**
- Seleccionar Hard Disk File Type and Format->VDI
-Seleccionar Use an Existing Virtual Hard Disk File->Path imagenes contruidas
