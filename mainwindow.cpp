#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "datasetmanager.h"
#include "debug.h"
#include "windowmanager.h"
#include "scriptmanager.h"
#include "common.h"

#include "scripting/scriptedit.h"

#include "widgets/imagewidget.h"
#include "widgets/userscriptwidget.h"

#include "vision/pyramids.h"

#include "opencv/opencv.h"

#include <QListView>
#include <QSettings>
#include <QCompleter>
#include <QMetaMethod>
#include <QVBoxLayout>
#include <QInputDialog>
#include <QScriptEngine>
#include <QDesktopWidget>
#include <QStringListModel>

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

	QSettings sets;

	ScriptManager sm;
};

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow),
	p(new MainWindowPriv)
{
	ui->setupUi(this);

	QDesktopWidget *d = QApplication::desktop();
	QRect r = d->screenGeometry(1);
	move(r.left(), 50);

	p->edit = new ScriptEdit(ui->frameScript);
	p->edit->setFocus();
	p->edit->setHistory(p->sets.value("history").toStringList());
	ui->listHistory->addItems(p->sets.value("history").toStringList());
	ui->listHistory->scrollToBottom();
	ui->frameScript->setLayout(new QVBoxLayout());
	ui->frameScript->layout()->addWidget(p->edit);
	connect(p->edit, SIGNAL(newEvaluation(QString)), SLOT(scriptTextChanged(QString)));

	p->edit->insertCompletion("iw", getObjectCompletions(ImageWidget::staticMetaObject));
	p->edit->insertCompletion("cv", getObjectCompletions(OpenCV::staticMetaObject));
	p->edit->insertCompletion("cmn", getObjectCompletions(Common::staticMetaObject));
	p->edit->insertCompletion("dm", getObjectCompletions(DatasetManager::staticMetaObject));
	p->edit->insertCompletion("pyr", getObjectCompletions(Pyramids::staticMetaObject));
	p->edit->insertCompletion("wm", getObjectCompletions(WindowManager::staticMetaObject));
	p->edit->insertCompletion("sm", getObjectCompletions(ScriptManager::staticMetaObject));

	p->sm.evaluateScript(p->sets.value("autostart").toString());
	activateWindow();
	p->edit->setFocus();
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pushEvaluate_clicked()
{
	QString str = ui->textBatch->toPlainText();
	p->sm.evaluateScript(str);
	activateWindow();
	p->edit->setFocus();
}

void MainWindow::scriptTextChanged(const QString &text)
{
	if (text.isEmpty())
		return;
	QStringList h = p->sets.value("history").toStringList();
	h << text;
	p->sets.setValue("history", h);
	ui->listHistory->addItem(text);
	p->sm.evaluateScript(text);
	activateWindow();
	p->edit->setFocus();
}

void MainWindow::on_actionInit_Commands_triggered()
{
	QString text = p->sets.value("autostart").toString();
	QString newtext = QInputDialog::getMultiLineText(this, trUtf8("Initializing commands"), trUtf8("Please type-in your autostart commands"), text);
	if (!newtext.isEmpty())
		p->sets.setValue("autostart", newtext);
}

void MainWindow::on_listHistory_itemDoubleClicked(QListWidgetItem *item)
{
	if (!item || item->text().isEmpty())
		return;
	p->edit->insertPlainText(item->text().trimmed());
	QApplication::sendEvent(p->edit, new QKeyEvent(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier));
}

void MainWindow::on_actionScripts_Editor_triggered()
{
	UserScriptWidget w;
	//w.setWindowModality(Qt::ApplicationModal);
	w.show();
	while (w.isVisible())
		QApplication::processEvents();
	if (!w.runText.isEmpty())
		p->sm.evaluateScript(w.runText);
	activateWindow();
	p->edit->setFocus();
}

void MainWindow::closeEvent(QCloseEvent *ev)
{
	QMainWindow::closeEvent(ev);
	ev->accept();
	QApplication::closeAllWindows();
}
