#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "datasetmanager.h"
#include "debug.h"
#include "windowmanager.h"
#include "scriptmanager.h"
#include "common.h"

#include "scripting/scriptedit.h"

#include "widgets/imagewidget.h"

#include "vision/pyramids.h"

#include "opencv/opencv.h"

#include <QListView>
#include <QSettings>
#include <QCompleter>
#include <QMetaMethod>
#include <QVBoxLayout>
#include <QScriptEngine>
#include <QStringListModel>

static void addQObject(QScriptEngine &e, QObject *obj, const QString &oname)
{
	QScriptValue iv = e.newQObject(obj);
	e.globalObject().setProperty(oname, iv);
}

static const QStringList getObjectCompletions(const QMetaObject &obj)
{
	QStringList methods;
	for (int j = obj.methodOffset(); j < obj.methodCount(); j++) {
		if (obj.method(j).methodType() == QMetaMethod::Slot ||
				obj.method(j).methodType() == QMetaMethod::Method)
			methods << QString::fromLatin1(obj.method(j).methodSignature());
	}
	return methods;
}

class MainWindowPriv
{
public:
	MainWindowPriv()
		: sets("settings.ini", QSettings::IniFormat)
	{

	}

	ScriptEdit *edit;
	ImageWidget *image;
	DatasetManager *dm;
	QScriptEngine eng;
	Pyramids *pyr;
	QSettings sets;
	WindowManager wm;
	ScriptManager sm;
};

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow),
	p(new MainWindowPriv)
{
	ui->setupUi(this);

	p->edit = new ScriptEdit(ui->frameScript);
	p->edit->setFocus();
	p->edit->setHistory(p->sets.value("history").toStringList());
	ui->frameScript->setLayout(new QVBoxLayout());
	ui->frameScript->layout()->addWidget(p->edit);
	connect(p->edit, SIGNAL(newEvaluation(QString)), SLOT(scriptTextChanged(QString)));

	p->image = new ImageWidget(ui->frameImage);
	ui->frameImage->setLayout(new QVBoxLayout());
	ui->frameImage->layout()->addWidget(p->image);

	p->dm = new DatasetManager(this);
	p->dm->addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/work/oxq1/oxbuild_images/");

	p->pyr = new Pyramids(this);

	p->sm.setScriptEngine(&p->eng);
	p->sm.setWindowManager(&p->wm);

	addScriptObject(p->dm, "dm");
	addScriptObject(p->image, "iw");
	addScriptObject(p->pyr, "pyr");
	addScriptObject(&p->wm, "wm");
	addScriptObject(&p->sm, "sm");

	/* find completions */
	QHashIterator<QString, QObject *> i(scriptObjects);
	while (i.hasNext()) {
		i.next();
		QObject *obj = i.value();
		const QMetaObject *mobj = obj->metaObject();
		QStringList methods;
		for (int j = mobj->methodOffset(); j < mobj->methodCount(); j++) {
			if (mobj->method(j).methodType() == QMetaMethod::Slot ||
					mobj->method(j).methodType() == QMetaMethod::Method)
				methods << QString::fromLatin1(mobj->method(j).methodSignature());
		}
		if (methods.size())
			p->edit->insertCompletion(i.key(), methods);
		addQObject(p->eng, i.value(), i.key());
	}
	p->edit->insertCompletion("iw", getObjectCompletions(ImageWidget::staticMetaObject));
	p->edit->insertCompletion("cv", getObjectCompletions(OpenCV::staticMetaObject));
	p->edit->insertCompletion("cmn", getObjectCompletions(Common::staticMetaObject));
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pushEvaluate_clicked()
{
}

void MainWindow::scriptTextChanged(const QString &text)
{
	if (text.isEmpty())
		return;
	QStringList h = p->sets.value("history").toStringList();
	h << text;
	p->sets.setValue("history", h);
	p->eng.evaluate(text);
	if (p->eng.hasUncaughtException())
		qDebug() << p->eng.uncaughtExceptionLineNumber() << p->eng.uncaughtException().toString();
	activateWindow();
	p->edit->setFocus();
}

void MainWindow::addScriptObject(QObject *obj, const QString &name)
{
	scriptObjects.insert(name, obj);
}
