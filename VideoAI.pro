#-------------------------------------------------
#
# Project created by QtCreator 2015-06-25T13:52:20
#
#-------------------------------------------------

QT       += core gui script scripttools network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VideoAI
TEMPLATE = app

DEFINES += DEBUG

include(build_config.pri)

SOURCES += main.cpp \
    scripting/scriptedit.cpp \
    scripting/scripthighlighter.cpp \
    scripting/tabsettings.cpp \
    scripting/textedit.cpp \
    widgets/imagewidget.cpp \
    datasetmanager.cpp \
    vision/pyramids.cpp \
    opencv/opencv.cpp \
    windowmanager.cpp \
    scriptmanager.cpp \
    common.cpp \
    vlfeat/vlfeat.cpp \
    snippets.cpp \
    svm/liblinear.cpp \
    svm/linear.cpp \
    svm/tron.cpp \
    vision/pyramidsvl.cpp \
    imps/oxfordretrieval.cpp \
    imps/caltechbench.cpp \
    caffe/caffecnn.cpp \
    lmm/buffercloner.cpp \
    lmm/bowpipeline.cpp \
    lmm/cnnpipeline.cpp \
    lmm/pipelinesettings.cpp \
    lmm/videopipeline.cpp \
    lmm/qtvideooutput.cpp \
    darknet_helper.c \
    lmm/vladpipeline.cpp

HEADERS  += \
    scripting/scriptedit.h \
    scripting/scripthighlighter.h \
    scripting/tabsettings.h \
    scripting/textedit.h \
    widgets/imagewidget.h \
    datasetmanager.h \
    vision/pyramids.h \
    opencv/opencv.h \
    windowmanager.h \
    scriptmanager.h \
    common.h \
    vlfeat/vlfeat.h \
    snippets.h \
    svm/liblinear.h \
    svm/linear.h \
    svm/tron.h \
    vision/pyramidsvl.h \
    imps/oxfordretrieval.h \
    imps/caltechbench.h \
    caffe/caffecnn.h \
    lmm/buffercloner.h \
    lmm/bowpipeline.h \
    lmm/cnnpipeline.h \
    lmm/lmmelements.h \
    lmm/pipelinesettings.h \
    lmm/videopipeline.h \
    lmm/qtvideooutput.h \
    darknet_helper.h \
    lmm/vladpipeline.h

INCLUDEPATH += .

RESOURCES += \
    scripting/images.qrc

CONFIG += opencv2 vlfeat lmm caffe cuda ui ffmpeg darknet

ui { include(widgets/widgets.pri) }

opencv2 {
    INCLUDEPATH += /usr/include/opencv
    LIBS += -lopencv_core \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lopencv_ml \
        -lopencv_video \
        -lopencv_features2d \
        -lopencv_calib3d \
        -lopencv_objdetect \
        -lopencv_contrib \
        -lopencv_legacy \
        -lopencv_flann \
        -lopencv_nonfree
}

opencv3 {
    INCLUDEPATH += /home/amenmd/myfs/source-codes/oss/build_x86/usr/local/include/
    LIBS += -L/home/amenmd/myfs/source-codes/oss/build_x86/usr/local/lib/
    LIBS += -lopencv_core \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_highgui \
        -lopencv_ml \
        -lopencv_video \
        -lopencv_features2d \
        -lopencv_calib3d \
        -lopencv_objdetect \
        -lopencv_flann \
        -lopencv_cudaarithm \
        -lopencv_xfeatures2d \
}

openmp {
    QMAKE_CXXFLAGS += -fopenmp
    LIBS += -fopenmp
}

vlfeat {
    DEFINES += HAVE_VLFEAT
    INCLUDEPATH += $$VLFEAT_PATH
    LIBS += -L$$VLFEAT_PATH/bin/glnxa64 -lvl
}

lmm {
    include($$INSTALL_PREFIX/usr/local/include/lmm/lmm.pri)
    DEFINES += HAVE_LMM
    SOURCES += lmm/classificationpipeline.cpp opencv/cvbuffer.cpp
    HEADERS += lmm/classificationpipeline.h opencv/cvbuffer.h

    LIBS += -lX11 -lXv
} else {
    SOURCES += debug.cpp
    HEADERS += debug.h
}

caffe {
    INCLUDEPATH += $$CAFFE_PATH/include/
    LIBS += -L$$CAFFE_PATH/lib -lcaffe
    LIBS += -lglog -lgflags -lprotobuf
    LIBS += -lboost_system -lboost_thread -lhdf5 -lhdf5_cpp -lhdf5_hl
    LIBS += -L/usr/lib64/atlas -llmdb
    #-lsatlas
    DEFINES += HAVE_CAFFE
}

darknet {
    INCLUDEPATH += /home/amenmd/myfs/source-codes/oss/darknet/install/include/
    LIBS += /home/amenmd/myfs/source-codes/oss/darknet/install/lib/libdarknet.a
    DEFINES += HAVE_DARKNET
    SOURCES += darknet.cpp
    HEADERS += darknet.h
    DEFINES += GPU OPENCV
}

cuda {
    INCLUDEPATH += /usr/local/cuda/include
    LIBS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
}

ffmpeg {
    #LIBS += -L -lavcodec -lavformat -lavutil
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libavformat.a
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libavcodec.a
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libavdevice.a
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libavutil.a
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libswresample.a
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libswresample.a
    LIBS += /home/amenmd/myfs/source-codes/oss/ffmpeg/ffmpeg/ffmpeg_build/lib/libswscale.a
    LIBS += /usr/lib/x86_64-linux-gnu/libvorbis.a
    LIBS += /usr/lib/x86_64-linux-gnu/libvorbisfile.a
    LIBS += /usr/lib/x86_64-linux-gnu/libvorbisenc.a
    LIBS += /usr/lib/x86_64-linux-gnu/libtheora.a
    LIBS += /usr/lib/x86_64-linux-gnu/libtheoraenc.a
    LIBS += /usr/lib/x86_64-linux-gnu/libopus.a
    LIBS += /usr/lib/x86_64-linux-gnu/libmp3lame.a
    LIBS += -lz -llzma -lx264 -lva -logg
}

LIBS += -L/usr/lib/libblas -lblas

OTHER_FILES += \
    docs/pipelines.txt
