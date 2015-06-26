#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "datasetmanager.h"
#include "debug.h"
#include "windowmanager.h"

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

Q_DECLARE_METATYPE(Mat)

static void addQObject(QScriptEngine &e, QObject *obj, const QString &oname)
{
	QScriptValue iv = e.newQObject(obj);
	e.globalObject().setProperty(oname, iv);
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
};

QScriptValue matToScriptValue(QScriptEngine *eng, const Mat &m)
{
	OpenCV *cvmat = new OpenCV(m);
	QScriptValue obj = eng->newQObject(cvmat, QScriptEngine::ScriptOwnership);
	///QScriptValue obj = eng->newObject();
	obj.setProperty("rows", m.rows);
	obj.setProperty("cols", m.cols);
	return obj;
}

void scriptToMat(const QScriptValue &obj, Mat &m)
{
	if (obj.isObject()) {
		OpenCV *cvmat = (OpenCV *)obj.toQObject();
		m = cvmat->getRefMat();
	}
	//m.rows = obj.property("rows").toInt32();
	//m.cols = obj.property("cols").toInt32();
}

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow),
	p(new MainWindowPriv)
{
	ui->setupUi(this);

	qScriptRegisterMetaType(&p->eng, matToScriptValue, scriptToMat);

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

	addScriptObject(p->dm, "dm");
	addScriptObject(p->image, "iw");
	addScriptObject(p->pyr, "pyr");
	addScriptObject(&p->wm, "wm");

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
	}

	QHashIterator<QString, QObject *> j(scriptObjects);
	while (j.hasNext()) {
		j.next();
		addQObject(p->eng, j.value(), j.key());
	}
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
}

void MainWindow::addScriptObject(QObject *obj, const QString &name)
{
	scriptObjects.insert(name, obj);
}
