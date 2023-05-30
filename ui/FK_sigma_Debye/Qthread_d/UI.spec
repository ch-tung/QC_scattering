# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['start.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['asn1crypto','certifi','chardet','conda','cryptography','idna','mccabe','pip','pyOpenSSL','PySocks','python-dateutil','urllib3','wincertstore','gpflow','imageio','imageio-ffmpeg','jupyter','jupyterlab','keras','Keras-Applications','keras-nightly','Keras-Preprocessing','meshio','numba','scikit-image','scikit-learn','smop','Sphinx','spyder','spyder-kernels','tensorboard','tensorboard-data-server','tensorboard-plugin-wit','tensorflow','tensorflow-estimator','tensorflow-gpu','tensorflow-io-gcs-filesystem','tensorflow-probability','trame'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FK_sigma_Debye',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
