#include "userscriptwidget.h"
#include "ui_userscriptwidget.h"
#include "debug.h"
#include "common.h"

#include <QDir>
#include <QInputDialog>

UserScriptWidget::UserScriptWidget(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::UserScriptWidget)
{
	ui->setupUi(this);

	QDir::current().mkdir("scripts");
	QStringList scripts = Common::listDir("scripts", "txt");
	ui->listScripts->addItems(scripts);
}

UserScriptWidget::~UserScriptWidget()
{
	delete ui;
}

void UserScriptWidget::on_pushSave_clicked()
{
	QString text;
	if (ui->listScripts->currentItem())
		text = ui->listScripts->currentItem()->text();
	QString name = QInputDialog::getText(this, trUtf8("Script name"), trUtf8("Please enter new scripts name"), QLineEdit::Normal, text);
	if (name.isEmpty())
		return;
	name.remove(".txt");
	if (!name.startsWith("scripts/"))
		name.prepend("scripts/");
	Common::exportText(ui->textCode->toPlainText().toUtf8(), QString("%1.txt").arg(name));
}

void UserScriptWidget::on_pushRun_clicked()
{
	runText = ui->textCode->toPlainText();
	close();
}

void UserScriptWidget::on_listScripts_currentRowChanged(int currentRow)
{
	if (currentRow < 0)
		return;
	ui->textCode->setPlainText(Common::importText(ui->listScripts->item(currentRow)->text()).join("\n"));
}

void UserScriptWidget::on_listScripts_itemDoubleClicked(QListWidgetItem *item)
{
	Q_UNUSED(item);
	/* currentRowChanged is already called by now */
	on_pushRun_clicked();
}
