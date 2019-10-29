/************************************************************
                    MAC449 - SSSKEL

 "Deteccao de bordas usando o algoritmo Tracking Contour"	  
 
 Autores: Emerson Luiz Navarro Tozette 

 Copyright (C) 2002 Emerson Luiz Navarro Tozette

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor,
 Boston, MA 02110-1301, USA.

**************************************************************/

#include <stdio.h>
#include <stdlib.h>
/* #include <errno.h> */
#include <string.h>
#include "mex.h"

/* CONSTANTES E STRUCTS: */

/* MAX: o contorno de uma imagem sera salvo em um vetor de 
   MAX posicoes. */

/* #define MAX 100000 */

/* struct pixel: essa struct tera dois campos. Esses campos
   formam a coordenada do pixel de uma imagem. */

struct pixel {

  int i, j;  /* i: linha, j: coluna */

};

/* lista ligada: */

struct node {
  struct pixel p;
  struct node * next;
};

typedef struct node Node;

static int llCONT = 0;
static Node * llINI = NULL, * llFIM = NULL;


Node *
llAlloc () {

  Node * n = (Node *) mxMalloc (sizeof(Node));
  if ( n == NULL ) {
    mexErrMsgTxt ("llAlloc: memoria insuficiente: exit(1)\n");
    /* exit (1); */
  }
  return n;
}

Node *
llBegin () {
 
  return llINI;
}

Node *
llAdd ( struct pixel p ) {

  if ( llINI == NULL ) { /* se lista vazia... */
    llINI = llAlloc();
    llFIM = llINI;
    /* llINI->p = p; */
    /* llINI->next = NULL; */
  }
  else {
    llFIM->next = llAlloc();
    llFIM = llFIM->next;
    /* llFIM->p = p; */
    /* llFIM->next = NULL; */
  }
  llFIM->p = p;
  llFIM->next = NULL;
  llCONT++;
  return llFIM;
}

/* matriz de marcacao de contornos: */

struct mark {
	int a, b, c, d;
};

struct mark * * mMatriz; /* minha matriz de marcacao */

/* verifica se rotulo pertence a posicao mMatriz[i][j]: */
int mMarcado (int i, int j, int rotulo) { 

	if(mMatriz[i][j].a == rotulo) {
		return 1;
	}
	if(mMatriz[i][j].b == rotulo) {
		return 1;
	}
	if(mMatriz[i][j].c == rotulo) {
		return 1;
	}
	if(mMatriz[i][j].d == rotulo) {
		return 1;
	}
	return 0;
}

void mAdd (int i, int j, int rotulo) { /* adiciona o rotulo em mMatriz[i][j] */

	if(mMarcado(i,j,rotulo)) return; /* se o rotulo ja foi marcado eu retorno. */

	if(mMatriz[i][j].a == 0) {
		mMatriz[i][j].a = rotulo;
		return;
	}
	if(mMatriz[i][j].b == 0) {
		mMatriz[i][j].b = rotulo;
		return;
	}
	if(mMatriz[i][j].c == 0) {
		mMatriz[i][j].c = rotulo;
		return;
	}
	if(mMatriz[i][j].d == 0) {
		mMatriz[i][j].d = rotulo;
		return;
	}
	mexPrintf("mAdd: %d, %d, %d, %d; rotulo: %d\n",
	mMatriz[i][j].a, mMatriz[i][j].b, mMatriz[i][j].c, mMatriz[i][j].d, rotulo); 
	/* mexErrMsgTxt("Erro em mAdd: exit(1)\n"); */
}	



/* FUNCOES: */

/* 1) DIFF: recebe dois pixels A e B, e retorna 1 caso as coordenadas
   desses pixels sejam diferentes ( e 0 caso contrario ): */

int 
DIFF ( struct pixel A, struct pixel B ) {

  if ( A.i != B.i || A.j != B.j ) return 1;
  else return 0;

}

/* 2) white: recebe um inteiro p e devolve 1 caso p seja igual a zero,
   ou devolve 0 caso p seja igual a um. A ideia eh que p seja o valor
   de algum pixel de uma imagem binaria. */

int
isBg (int p) { /* white ( int p ) { */

  if ( p <= 0 ) return 1; /* bg */
  else return 0; /* objeto */
  
}

static int OBJ_LABEL;

int 
isObj (int p) {

  if ( p == OBJ_LABEL ) return 1;
  else return 0;

}

/* 3) Encontra_prim_pixel: recebe uma matriz de inteiros (imagem) de 
   dimensao N X M, e devolve o 'starting pixel' do contorno dessa imagem :
*/

/********* eliminado *******/

/* 4) Encontra_segundo_pixel:  recebe uma matriz de inteiros (img) de
   dimensao N X M e o 'starting pixel' P do contorno de img, e devolve
   o proximo pixel p desse contorno e a direcao dcn (direcao do pixel
   P para o pixel p). O pixel p eh o primeiro pixel candidato encontrado.
   dcn eh uma direcao do codigo chain code. */

void 
Encontra_segundo_pixel ( int * * img, int N, int M, struct pixel P,
			 struct pixel * p, int * dcn ) {

  int i, j;
  i = P.i; j = P.j;

  if ( isBg(img[i][j-1]) && isObj(img[i+1][j-1])/* && white(mark[i][j-1])*/ /* <= 1*/) {
    p->i = i;
    p->j = j-1;
    *dcn = 4;
    /* printf("E.2.Pixel: %d %d\n", i, j-1); */
    return;
  }
  if ( isBg(img[i+1][j-1]) && isObj(img[i+1][j])/* && white(mark[i+1][j-1])*/ /* <= 1*/) {
    p->i = i+1;
    p->j = j-1;
    *dcn = 5;
    /*printf("E.2.Pixel: %d %d\n", i+1, j-1);*/
    return;
  }
  if ( isBg(img[i+1][j]) && isObj(img[i+1][j+1])/* && white(mark[i+1][j])*/ /* <= 1*/) {
    p->i = i+1;
    p->j = j;
    *dcn = 6;
    /*printf("E.2.Pixel: %d %d\n", i+1, j);*/
    return;
  }
  if ( isBg(img[i+1][j+1]) && isObj(img[i][j+1])/* && white(mark[i+1][j+1])*/ /* <= 1*/) {
    p->i = i+1;
    p->j = j+1;
    *dcn = 7;
    /*printf("E.2.Pixel: %d %d\n", i+1, j+1);*/
    return;
  }

  /*printf("E.2.Pixel:ihhh\n");*/
  p->i = -1;
  p->j = -1;
  return;
}

/* 5) invert: recebe uma direcao d (do chain code) e devolve a direcao
   oposta a essa direcao. */


int 
invert ( int d ) {

  switch (d) {

  case 0: return 4;
  case 1: return 5;
  case 2: return 6;
  case 3: return 7;
  case 4: return 0;
  case 5: return 1;
  case 6: return 2;
  case 7: return 3;
  default: mexErrMsgTxt ("\nErro: funcao invert: exit(1)\n");
    /*exit(1);*/
  }

}

/* 6) chainpoint: devolve o pixel vizinho ao pixel Pc que se encontra
   na direcao d ( do chain code). */


struct pixel
chainpoint ( struct pixel Pc, int d ) {

  int i, j;
  struct pixel p;
  
  i = Pc.i; j = Pc.j;
  
  switch (d) {

  case 0: p.i = i;    p.j = j+1;  break;
  case 1: p.i = i-1;  p.j = j+1;  break;
  case 2: p.i = i-1;  p.j = j;    break;
  case 3: p.i = i-1;  p.j = j-1;  break;
  case 4: p.i = i;    p.j = j-1;  break;
  case 5: p.i = i+1;  p.j = j-1;  break;
  case 6: p.i = i+1;  p.j = j;    break;
  case 7: p.i = i+1;  p.j = j+1;  break;
  }
  
  return p;


}


/* 7) find_next: esta funcao eh usada para encontrar todos os pixels
   do contorno de uma imagem (img) que sejam diferentes do 'starting pixel'
   e do segundo pixel do contorno dessa imagem (esses pixels sao
   encontados pelas funcoes Encontra_prim_pixel e Encontra_segundo_pixel,
   respectivamente). Ela recebe uma matriz img, um pixel Pc (current pixel),
   a direcao dpc ( direcao do 'previous pixel' para o 'current pixel'. Previous
   pixel eh o pixel imediatamente anterior ao current pixel segundo o contorno
   da imagem), e devolve Pn (o 'next pixel') e dcn (direcao current->next).
   Todas as direcoes se referem ao chain code. */

void
find_next (struct pixel Pc, int dpc, struct pixel * Pn, int * dcn,
	   int * * img ) {

  int dcp, de, di, r;
  struct pixel Pe, Pi;
  Pn->i = -1;
  Pn->j = -1;
  dcp = invert (dpc);
  for ( r = 0; r </*=*/ 7; r++ ) {   
    /* printf("r: %d\n", r); */
    de = (dcp+r) % 8;
    di = (dcp+r+1) % 8;
    Pe = chainpoint ( Pc, de );
    Pi = chainpoint ( Pc, di );
    if ( isBg(img[Pe.i][Pe.j]) && isObj(img[Pi.i][Pi.j])/*&& white(mark[Pe.i][Pe.j])*/ /* <= 1*/) {
      *Pn = Pe;
      *dcn = de;
      /* printf("OK:%d %d\n", Pe.i, Pe.j); */
    }
  }
  /* printf("Find-Next: ihhhh\n"); */

}


/* 8) Contour: recebe uma matriz de inteiros (img) de dimensao N X M e um
   vetor de pixels E[1..MAX-1] e salva o contorno dessa imagem em E.
   Retorna o numero de pixels+1 do contorno da imagem. */

/* int ** IMAGEM_FINAL; */

/* char *idxx; */

int
Contour ( struct pixel P1, int * * img, int N, int M ) { /* P1:starting pixel */

  /* int I,J; */
  /* FILE * arq_imagem; */
  /* int i; */
  int n, dcn, dpc;
  struct pixel next;

  /* C = fopen("CONT.txt", "a"); */

  /*E[1] = Encontra_prim_pixel (img, N, M);*/
  /*if (E[1].i == -1 || E[1].j == -1)*/
  /*return (-1);*/
  
  n = 2;
  Encontra_segundo_pixel (img, N, M, /*E[1]*/P1, &next, &dcn);
  if (next.i == -1 || next.j == -1)
    return n;
  
  while ( DIFF ( next, /*E[1]*/ P1 ) ) {
    if (n > 200000) {
      mexPrintf("ESTOUROU\n");
      break;
    }
   
    /* E[n] = next; */
    llAdd (next);
    
    dpc = dcn;
    find_next ( /*E[n]*/ next, dpc, &next, &dcn, img );
    if (next.i == -1 || next.j == -1) {
      mexPrintf("Find-Next: ihhhh\n");
      break;
    }
    n = n + 1;
  }

  return n;
  
}
    

/*** Dilatacao Exata e Esqueleto (imagem diferenca): ***/

/* 1) cria_SEDR(): recebe um numero inteiro Nm, que eh o numero de dilatacoes
   exatas que pretendemos efetuar para cada pixel do contorno (ou contornos)
   da nossa imagem, e retorna um apontador para um vetor de pixels. Esse
   vetor eh a nossa estrutura auxiliar de dados SEDR (sorted Euclidean distance
   representation). 
  		Para cada pixel P de uma imagem, existem 8 ou 4 pixels que se encontram
   a uma distancia exata d. Esses pixels sao simetricos em relacao a P. A funcao
   abaixo encontra apenas um pixel P' desses 8 (ou 4) pixels, e salva a posicao de P'
   relativa a P no vetor SEDR. Os outros 7 (ou 3) pixels devem ser encontrados 
   refletindo P' nos eixos de simetria de P (isso eh feito pela funcao dilata() ); */

struct pixel *
cria_SEDR ( int Nm ) {
  
  int i, j, k = 0, b = 0;
  struct pixel * SEDR;
  SEDR = (struct pixel *) mxMalloc (sizeof(struct pixel) * (Nm+1));
  
  /* A posicao SEDR[b] guarda a posicao de P' relativa a P na distancia b,
     b == 0..Nm . Assume-se que P esta na coordenada (0,0) do plano cartesiano 
     xy. */
  for ( i = 0; ; i++ ) {
    for ( j = 0; j <= k ; j++ )	{
      SEDR[b].i = j; /* SEDR[b].i == coordenada de P' no eixo x */
      SEDR[b].j = i; /* SEDR[b].j == coordenada de P' no eixo y */
      b++;
      if (b > Nm) return SEDR;
    }
    k++;
  }
}

/* 2) dentro(): retorna 1 se n == 0..N-1 e m == 0..M-1, retorna 0 caso contrario. */

int dentro (int n, int m, int N, int M) {

  if( 0 <= n && n < N && 0 <= m && m < M) return 1;
  else return 0;
}

/* 3) dilata(): dilata os pixels contidos no vetor contorno[1..r] ate a distancia
   exata Nm, usando a informacao contida na estrutura SEDR. 
  		O contorno (ou contornos) dilatados sao salvos na imagem img_dil, de 
   dimensoes N x M.
        Cada pixel da imagem resultante img_dil foi obtido a partir da dilatacao
   de um determinado contorno de comprimento rr. A matriz maux (NxM) associa cada 
   pixel de img_dil ao comprimento rr. Essa informacao sera utilizada pela
   funcao Esqueleto().
  		Note que o vetor contorno[] contem todos os contornos obtidos da imagem
   de entrada do programa. */

void dilata( struct pixel * SEDR, int Nm, int ** img_dil, int N, int M) {
  /* struct pixel * contorno, int r ,int ** maux ) { */
  
  int lin, col, x, y;
  int j,ii;
  char buf[30];
  Node * ll;
  
  mexPrintf("\nENTRA-DILATA\n"); 
  
  for( j = 1/*0*/; j <= Nm; j++ ) {
    x = SEDR[j].i;
    y = SEDR[j].j;
    ii = 1;  /* <<<--- */
    for ( ll = llBegin(); ll != NULL; ll = ll->next ) {
      lin = ll->p.i;   /* contorno[i].i; */
      col = ll->p.j;   /* contorno[i].j; */
      
      /* Cada pixel do meu contorno recebe um rotulo ii:  <<<--- */
      if (j == 1) img_dil[lin][col] = ii; /* soh rotulo uma vez	*/
      
      /* Agora devo propagar os rotulos usando dilatacao exata. Note que o valor
         rr tambem eh propagado pela matriz maux.
         A propagacao eh feita refletindo-se o pixel (lin+x,col+y) sobre os eixos
         de simetria centrados em (lin,col). Cada pixel (lin,col) eh propagado 8
         (ou 4) vezes: */
      
      /* sprintf(buf,"(((%d,%d)\n",lin-y,col+x);mexPrintf(buf); */
      if( dentro(lin-y, col+x, N, M) ) 
	if( img_dil [lin-y][col+x] == 0 ) { img_dil [lin-y][col+x] = ii; /*  maux[lin-y][col+x]=rr; */}
      
      /* sprintf(buf,"(((%d,%d)\n",lin+y,col+x);mexPrintf(buf); */
      if( dentro(lin+y, col+x, N, M) )
	if( img_dil [lin+y][col+x] == 0 ) { img_dil [lin+y][col+x] = ii; /*  maux[lin+y][col+x]=rr; */}

      /* sprintf(buf,"(((%d,%d)\n",lin-y,col-x);mexPrintf(buf); */
      if( dentro(lin-y, col-x, N, M) )
	if( img_dil [lin-y][col-x] == 0 ) { img_dil [lin-y][col-x] = ii;/*   maux[lin-y][col-x]=rr; */}
      
      /* sprintf(buf,"(((%d,%d)\n",lin+y,col-x);mexPrintf(buf); */
      if( dentro(lin+y, col-x, N, M) )
	if( img_dil [lin+y][col-x] == 0 ) { img_dil [lin+y][col-x] = ii; /*  maux[lin+y][col-x]=rr; */}
      
      
      /* sprintf(buf,"(((%d,%d)\n",lin-x,col+y);mexPrintf(buf); */
      if( dentro(lin-x, col+y, N, M) )
	if( img_dil [lin-x][col+y] == 0 ) { img_dil [lin-x][col+y] = ii; /*  maux[lin-x][col+y]=rr; */}

      /* sprintf(buf,"(((%d,%d)\n",lin+x,col+y);mexPrintf(buf); */
      if( dentro(lin+x, col+y, N, M) )
	if( img_dil [lin+x][col+y] == 0 ) { img_dil [lin+x][col+y] = ii; /*  maux[lin+x][col+y]=rr; */}
      
      /* sprintf(buf,"(((%d,%d)\n",lin-x,col-y);mexPrintf(buf); */
      if( dentro(lin-x, col-y, N, M) )
	if( img_dil [lin-x][col-y] == 0 ) { img_dil [lin-x][col-y] = ii; /*  maux[lin-x][col-y]=rr; */}
      
      /* sprintf(buf,"(((%d,%d)\n",lin+x,col-y);mexPrintf(buf); */
      if( dentro(lin+x, col-y, N, M) )
	if( img_dil [lin+x][col-y] == 0 ) { img_dil [lin+x][col-y] = ii; /*  maux[lin+x][col-y]=rr; */}
      
      /* if (ii == rr ) ii = 1; */ /* <<<--- */
      /* else ii++; */
      ii++;

    }/* end_for */
  }/* end_for */
  
  sprintf(buf,"EXIT-dilata - ii: %d\n",ii);
  mexPrintf(buf);
}

/* 4) Maximo(): retorna o maior valor de um vetor de inteiros v[0..n-1]. */

int Maximo( int * v, int n ) {
   
  int i;
  int m = 0;
  for ( i = 0; i < n; i++ )
    if( m < v[i]) m = v[i];
  
  return m;
}

/* 5) Esqueleto(): recebe as matrizes img_lbl (NxM) e maux (NxM), onde img_lbl
   eh a imagem com rotulos propagados atraves de dilatacao exata, e maux eh 
   a matriz que associa cada pixel P de img_lbl com o comprimento do contorno 
   que contem P (ambas as matrizes sao calculadas pela funcao dilata()). 
	Essa funcao calcula a imagem diferenca (img_dif) de img_lbl. 
	A informacao contida na matriz maux eh utilizada para tratar os choques
	entre os elementos perto das extremidades do rotulo de um contorno. */

void Esqueleto (int ** img_lbl, int N, int M, int ** img_dif, int R ) { /* ** maux) { */

  int x, y, max, v[4];/*, r; */
  /* char buf[40]; */

  mexPrintf("\nENTRA-Esqueleto\n");

  for( x = 1; x < N-1; x++ )
    for( y = 1; y < M-1; y++) {
      /* r = maux[x][y]; */
      v[0] = abs(img_lbl[x][y] - img_lbl[x+1][y]);	
      v[1] = abs(img_lbl[x][y] - img_lbl[x][y+1]);
      v[2] = abs(img_lbl[x][y] - img_lbl[x-1][y]);
      v[3] = abs(img_lbl[x][y] - img_lbl[x][y-1]);
      max = Maximo(v,4);
      if( max < R/2.0) img_dif[x][y] = max;
      else {
	img_dif[x][y] = abs(R - max);
	/* sprintf(buf," ESQ: r:%d (%.3f) max:%d r-max:%d abs(r-max):%d\n",r,r/2.0,max,r-max,abs(r-max)); */
	/* mexPrintf(buf); */
      }
    }
  mexPrintf("\nEXIT-Esqueleto\n");
}

/*******************************************************************************************/
/* -->> dilatacao e esqueleto usando rotulos dependentes e tratamento de choque individual
 */


/* 3) dilata2(): dilata os pixels contidos no vetor contorno[1..r] ate a distancia
   exata Nm, usando a informacao contida na estrutura SEDR. 
  		O contorno (ou contornos) dilatados sao salvos na imagem img_dil, de 
   dimensoes N x M.
        Cada pixel da imagem resultante img_dil foi obtido a partir da dilatacao
   de um determinado contorno de comprimento rr. A matriz maux (NxM) associa cada 
   pixel de img_dil ao comprimento rr. Essa informacao sera utilizada pela
   funcao Esqueleto2().
  		Note que o vetor contorno[] contem todos os contornos obtidos da imagem
   de entrada do programa. */

void dilata2( struct pixel * SEDR, int Nm, int ** img_dil, int N, int M,
	      /*struct pixel * contorno, int r ,*/ int ** maux ) { 
  
  int lin, col, x, y, rr;
  int j, ii;
  Node * ll;
  
  mexPrintf("ENTRA-DILATA2\n");
  
  for( j = 0; j <= Nm; j++ ) {
    x = SEDR[j].i;
    y = SEDR[j].j;
    /*ii = 1; */ /* <<<--- */
    for( ii = 1, ll = llBegin(); ll != NULL; ll = ll->next, ii++ ) {
      lin = ll->p.i;
      col = ll->p.j;
      
      /* Inicialmente, a matriz maux ja contem na posicao [lin][col] o comprimento
         rr do contorno que contem o pixel [lin][col]: */
      rr = maux[lin][col];
      
      /* Cada pixel do meu contorno recebe um rotulo ii:  <<<--- */
      if( img_dil [lin][col] == 0 ) img_dil[lin][col] = ii; /* <<<<----- */
      
      /* Agora devo propagar os rotulos usando dilatacao exata. Note que o valor
         rr tambem eh propagado pela matriz maux.
         A propagacao eh feita refletindo-se o pixel (lin+x,col+y) sobre os eixos
         de simetria centrados em (lin,col). Cada pixel (lin,col) eh propagado 8
         (ou 4) vezes: */
      
      /* sprintf(buf,"(((%d,%d)\n",lin-y,col+x);mexPrintf(buf); */
      if( dentro(lin-y, col+x, N, M) ) 
	if( img_dil [lin-y][col+x] == 0 ) { img_dil [lin-y][col+x] = ii; maux[lin-y][col+x]=rr;}
      
      /* sprintf(buf,"(((%d,%d)\n",lin+y,col+x);mexPrintf(buf); */
      if( dentro(lin+y, col+x, N, M) )
	if( img_dil [lin+y][col+x] == 0 ) { img_dil [lin+y][col+x] = ii; maux[lin+y][col+x]=rr;}

      /* sprintf(buf,"(((%d,%d)\n",lin-y,col-x);mexPrintf(buf); */
      if( dentro(lin-y, col-x, N, M) )
	if( img_dil [lin-y][col-x] == 0 ) { img_dil [lin-y][col-x] = ii; maux[lin-y][col-x]=rr;}
      
      /* sprintf(buf,"(((%d,%d)\n",lin+y,col-x);mexPrintf(buf); */
      if( dentro(lin+y, col-x, N, M) )
	if( img_dil [lin+y][col-x] == 0 ) { img_dil [lin+y][col-x] = ii; maux[lin+y][col-x]=rr;}
      
      
      /* sprintf(buf,"(((%d,%d)\n",lin-x,col+y);mexPrintf(buf); */
      if( dentro(lin-x, col+y, N, M) )
	if( img_dil [lin-x][col+y] == 0 ) { img_dil [lin-x][col+y] = ii; maux[lin-x][col+y]=rr;}

      /* sprintf(buf,"(((%d,%d)\n",lin+x,col+y);mexPrintf(buf); */
      if( dentro(lin+x, col+y, N, M) )
	if( img_dil [lin+x][col+y] == 0 ) { img_dil [lin+x][col+y] = ii; maux[lin+x][col+y]=rr;}
      
      /* sprintf(buf,"(((%d,%d)\n",lin-x,col-y);mexPrintf(buf); */
      if( dentro(lin-x, col-y, N, M) )
	if( img_dil [lin-x][col-y] == 0 ) { img_dil [lin-x][col-y] = ii; maux[lin-x][col-y]=rr;}
      
      /* sprintf(buf,"(((%d,%d)\n",lin+x,col-y);mexPrintf(buf); */
      if( dentro(lin+x, col-y, N, M) )
	if( img_dil [lin+x][col-y] == 0 ) { img_dil [lin+x][col-y] = ii; maux[lin+x][col-y]=rr;}
      
      /* if (ii == rr ) ii = 1; */ /* <<<--- */
      /* ii++; */


    }/* end_for */
  }/* end_for */
  
  mexPrintf("EXIT-dilata2\n");
}



/* 5) Esqueleto2(): recebe as matrizes img_lbl (NxM) e maux (NxM), onde img_lbl
   eh a imagem com rotulos propagados atraves de dilatacao exata, e maux eh 
   a matriz que associa cada pixel P de img_lbl com o comprimento do contorno 
   que contem P (ambas as matrizes sao calculadas pela funcao dilata()). 
  		Essa funcao calcula a imagem diferenca (img_dif) de img_lbl. 
  		A informacao contida na matriz maux eh utilizada para tratar os choques
   entre os elementos perto das extremidades do rotulo de um contorno. */

void Esqueleto2 (int ** img_lbl, int N, int M, int ** img_dif, int ** maux) {

   int x, y, max, v[4], r;
   
   mexPrintf("\nENTRA-Esqueleto2\n");

   for( x = 1; x < N-1; x++ )
     for( y = 1; y < M-1; y++) {
       r = maux[x][y];
       v[0] = abs(img_lbl[x][y] - img_lbl[x+1][y]);	
       v[1] = abs(img_lbl[x][y] - img_lbl[x][y+1]);
       v[2] = abs(img_lbl[x][y] - img_lbl[x-1][y]);
       v[3] = abs(img_lbl[x][y] - img_lbl[x][y-1]);
       max = Maximo(v,4);
       if( max < r/2.0) img_dif[x][y] = max;
       else {
	 img_dif[x][y] = abs(r - max);
	 /* sprintf(buf," ESQ: r:%d (%.3f) max:%d r-max:%d abs(r-max):%d\n",r,r/2.0,max,r-max,abs(r-max)); */
	 /* mexPrintf(buf); */
       }
     }
   mexPrintf("\nEXIT-Esqueleto2\n");
}


/* PROGRAMA PRINCIPAL: */

/* transf(): transforma a matriz a (m linhas e n colunas), que esta no formato 
   utilizado pelo Matlab (no Matlab matrizes sao guardadas por coluna em um 
   unico vetor), em uma matriz de inteiros no formato C: */

int ** transf (double * a, int m, int n) {
  int i, j, x;
  int ** r;
  char buf[20];

  mexPrintf(">>transf-BEG\n");
  
  r = (int **) mxMalloc (m * sizeof(int*));
  if (r == NULL) mexErrMsgTxt("transf: Falta de memoria\n");
  for(i = 0; i < m; i++) {
    r[i] =  (int *) mxMalloc (n * sizeof(int));
	if(r[i] == NULL) mexErrMsgTxt("transf: Falta de memoria\n");
        for (j = 0; j < n; j++){ /*sprintf(buf," %d %d %d\n", i,j,j*m+i);mexPrintf(buf);*/
      r[i][j] = (int) *(a + j * m + i); 
	}
  }
  mexPrintf(">>transf-END\n");
  return r;
}


/* int main ( int argc, char * argv[] ) { */
void 
mexFunction(int nlhs, mxArray *saida[], int nrhs, const mxArray *entrada[]) {


  /* 1) declaracao de variaveis: */

  /* int I, J; */
  /* FILE * arq_imagem; */ /* arquivo de entrada. */
  int i, j, tmp,k,       /* variaveis auxiliares. */
    M, N,            /* dimensao da matriz IMAGEM. */
    ** IMAGEM,       /* matriz que guardara o conteudo de arq_imagem. */
    ** IMAGEM_DIL, 
    ** IMAGEM_ESQ,
	** IMAGEM_CONT,
	** maux;
  struct pixel * SEDR ; 

  int l = 0, Nm;
  int r;

  struct pixel P1;
  Node * ll;

  double * imagem_entrada,
    * contorno,           /* (possuem o mesmo significado das matrizes acima). */
    * dilatacao, 
    * esqueleto;
  char buf[30];	

  /* dimensoes da imagem de entrada: */
  N = mxGetM(entrada[0]); /* linhas */
  M = mxGetN(entrada[0]); /* colunas */
  sprintf(buf, " NxM: %d * %d = %d\n", N, M, N*M); 
  mexPrintf(buf);
  
  if(!mxIsDouble(entrada[0])) 
	mexErrMsgTxt("A matriz de entrada nao eh do tipo double: exit(1)\n");
  
 
  /* 2) alocando as matrizes: */
  IMAGEM = (int **) mxMalloc ( N * sizeof (int *));
  IMAGEM_CONT = (int **) mxMalloc ( N * sizeof (int *));
  IMAGEM_DIL = (int **) mxMalloc ( N * sizeof (int *));
  IMAGEM_ESQ = (int **) mxMalloc ( N * sizeof (int *));
  maux = (int **) mxMalloc ( N * sizeof (int *));
  mMatriz =  (struct mark **) mxMalloc ( N * sizeof (struct mark *));
	 
  for ( i = 0; i < N; i++ ) {
    IMAGEM[i] =  (int *) mxMalloc ( M * sizeof (int));
    IMAGEM_CONT[i] =  (int *) mxMalloc ( M * sizeof (int));
    IMAGEM_DIL[i] =  (int *) mxMalloc ( M * sizeof (int));
    IMAGEM_ESQ[i] =  (int *) mxMalloc ( M * sizeof (int));
    maux[i] =  (int *) mxMalloc ( M * sizeof (int));
	mMatriz[i] =  (struct mark *) mxMalloc ( M * sizeof (struct mark));
  }

  imagem_entrada = mxGetPr(entrada[0]);   /* mexPrintf("oi\n"); */
  if(imagem_entrada == NULL) 
	mexErrMsgTxt("Erro em mxGetPr(entrada[0]): exit(1)\n");

  IMAGEM = transf (imagem_entrada, N, M);

  for ( i = 0; i < N; i++ )
    for ( j = 0; j < M; j++ ) {
      /*fscanf (arq_imagem,"%d", &IMAGEM[i][j]); */
      IMAGEM_CONT [i][j] = 0;
      IMAGEM_DIL [i][j] = 0;
      IMAGEM_ESQ [i][j] = 0;
      maux[i][j] = 0;
	  mMatriz[i][j].a = mMatriz[i][j].b = mMatriz[i][j].c = mMatriz[i][j].d = 0;
    }
  /*fclose (arq_imagem); */


  /* 3) detectando o contorno de IMAGEM: */

llCONT = 0;
llINI = llFIM = NULL;

  for ( i = 0; i <  N; i++ ) 
    for ( j = 0; j < M-1; j++ ) { /*ultima coluna ignorada*/
      if ( isBg(IMAGEM[i][j]) && !(isBg(IMAGEM[i][j+1])) ) {
		OBJ_LABEL = IMAGEM[i][j+1];
		if ( ! mMarcado(i,j,OBJ_LABEL) ) {  /*IMAGEM[i][j] != -OBJ_LABEL ) { */
			if(!(l%100)) {
				sprintf(buf,"l: %d\n", l);
				mexPrintf(buf); 
				
			}
			l++;
			/*E[1].i = i; */
			/*E[1].j = j; */
			P1.i = i; P1.j = j; ll = llAdd(P1); /* adiciona o primeiro pixel */
			r = Contour ( P1, IMAGEM, N, M );
			/*printf("main>> %d %d r:%d\n", E[1].i, E[1].j, r);*/
			/*for ( k = 1; k < r; k++ ) { */
			for ( ; ll != NULL; ll = ll->next ) {
			  IMAGEM_CONT [ll->p.i] [ll->p.j] = 1;  /* 1 == preto */
				/* IMAGEM [ll->p.i][ll->p.j] = -OBJ_LABEL; */
				mAdd (ll->p.i, ll->p.j, OBJ_LABEL);
				maux [ll->p.i][ll->p.j] = r-1; /* ou r? ou tanto faz? */
			}
		}
      }
    }
    
  mexPrintf("llCONT: %d\n", llCONT);

  /* dilatacao exata e esqueleto: */

  Nm = /*20;*/ (int) mxGetScalar(entrada[1]);
  mexPrintf("Nm: %d\n", Nm);
  SEDR = cria_SEDR ( Nm );
  /*dilata (SEDR, Nm, IMAGEM_DIL, N, M);*/
  /*Esqueleto (IMAGEM_DIL, N, M, IMAGEM_ESQ, llCONT );*/

  /* dilatacao e esqueleto 2: */

 /*for ( i = 0; i < N; i++ )
   for ( j = 0; j < M; j++ ) {
     IMAGEM_DIL [i][j] = 0;
     IMAGEM_ESQ [i][j] = 0;
   }*/ 

 dilata2 (SEDR, Nm, IMAGEM_DIL, N, M, maux );
 Esqueleto2 (IMAGEM_DIL, N, M, IMAGEM_ESQ, maux ); 

saida[0] = mxCreateDoubleMatrix (N, M, mxREAL);
if(saida[0] == NULL) mexErrMsgTxt("Erro em mxCreateDoubleMatrix \n");
saida[1] = mxCreateDoubleMatrix (N, M, mxREAL);  
if(saida[1] == NULL) mexErrMsgTxt("Erro em mxCreateDoubleMatrix \n");
saida[2] = mxCreateDoubleMatrix (N, M, mxREAL);  
if(saida[2] == NULL) mexErrMsgTxt("Erro em mxCreateDoubleMatrix \n");

 
 contorno = (double *) mxGetPr(saida[0]);
 dilatacao = (double *) mxGetPr(saida[1]);
 esqueleto = (double *) mxGetPr(saida[2]);
 /* 6) as imagens de saida sao transformadas em um vetor e passadas
    para o Matlab: */

  tmp = 0;
  for ( j = 0; j < M; j++){
    for ( k = 0; k < N; k++ ) {
      contorno[tmp]  = IMAGEM_CONT[k][j];
	  dilatacao[tmp]    = IMAGEM_DIL[k][j];
      esqueleto[tmp] = IMAGEM_ESQ[k][j];
      tmp++;
    }
  }

  mxSetPr(saida[0], contorno);
  mxSetPr(saida[1], dilatacao);
  mxSetPr(saida[2], esqueleto);

 return ;

}

