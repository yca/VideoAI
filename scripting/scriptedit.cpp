/****************************************************************************
**
** Copyright (C) 2007-2008 Trolltech ASA. All rights reserved.
**
** This file is part of the Qt Script Debug project on Trolltech Labs.
**
** This file may be used under the terms of the GNU General Public
** License version 2.0 as published by the Free Software Foundation
** and appearing in the file LICENSE.GPL included in the packaging of
** this file.  Please review the following information to ensure GNU
** General Public Licensing requirements will be met:
** http://www.trolltech.com/products/qt/opensource.html
**
** If you are unsure which license is appropriate for your use, please
** review the following information:
** http://www.trolltech.com/products/qt/licensing.html or contact the
** sales department at sales@trolltech.com.
**
** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
**
****************************************************************************/

#include "scriptedit.h"

#include "scripthighlighter.h"

#include <QStyle>
#include <QtGui/QPainter>
#include <QtCore/QTimer>
#include <QtCore/QDebug>

#include <QCompleter>
#include <QListView>
#include <QStringListModel>

ScriptEdit::ScriptEdit(QWidget *parent)
	: TextEdit(parent)
{
	highlighter =  new ScriptHighlighter(this);
	popup = new QListView();

	model = new QStringListModel(this);
	completer = new QCompleter();
	completer->setModel(model);
	completer->setCaseSensitivity(Qt::CaseInsensitive);
	completer->setCompletionMode(QCompleter::PopupCompletion);
	completer->setModelSorting(QCompleter::CaseSensitivelySortedModel);
	completer->setPopup(popup);
	completer->setWidget(this);

	histPos = -1;

	insertPlainText("$>");
}

ScriptEdit::~ScriptEdit()
{
}

void ScriptEdit::insertCompletion(const QString &name, const QStringList &list)
{
	completions.insert(name, list);
}

void ScriptEdit::keyPressEvent(QKeyEvent *ev)
{
	bool prompt = false;
	if (popup->isVisible()) {
		if (ev->key() == Qt::Key_Escape)
			popup->hide();
		else if (ev->key() == Qt::Key_Enter ||
				 ev->key() == Qt::Key_Return) {
			QTextCursor c = textCursor();
			QString text = model->data(popup->currentIndex(), Qt::DisplayRole).toString();
			c.insertText(text.split("(").first());
			c.insertText("(");
			setTextCursor(c);
		}
		ev->ignore();
		return;
	} else if (ev->key() == Qt::Key_Period) {
		QTextCursor c = textCursor();
		QString target = c.block().text().split(QRegExp("\\W"), QString::SkipEmptyParts).last();
		if (completions.contains(target)) {
			model->setStringList(completions[target]);
			completer->complete();
		}
	} else if (ev->key() == Qt::Key_Enter || ev->key() == Qt::Key_Return) {
		prompt = true;
		commandEntered(textCursor().block().text().remove("$>"));
	} else if (ev->key() == Qt::Key_Up) {
		if (histPos <= 0 || histPos >= history.size())
			histPos = history.size() - 1;
		else
			histPos--;
		replaceCurrentLine(history[histPos]);
		ev->ignore();
		return;
	} else if (ev->key() == Qt::Key_Down) {
		if (histPos >= history.size() - 1 || histPos < 0)
			histPos = 0;
		else
			histPos++;
		replaceCurrentLine(history[histPos]);
		ev->ignore();
		return;
	}
	TextEdit::keyPressEvent(ev);
	if (prompt)
		insertPrompt();
}

void ScriptEdit::commandEntered(const QString &line)
{
	if (line.isEmpty())
		return;
	history << line;
	histPos = -1;
	emit newEvaluation(line);
}

void ScriptEdit::insertPrompt()
{
	insertPlainText("$>");
}

void ScriptEdit::replaceCurrentLine(const QString &text)
{
	QTextCursor c = textCursor();
	moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
	moveCursor(QTextCursor::StartOfLine, QTextCursor::MoveAnchor);
	moveCursor(QTextCursor::End, QTextCursor::KeepAnchor);
	textCursor().removeSelectedText();
	setTextCursor(c);
	insertPrompt();
	c.insertText(text);
}

