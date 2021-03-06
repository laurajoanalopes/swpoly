/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "pair_swpoly.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairSWPOLY::PairSWPOLY(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  nelements = 0;
  elements = NULL;
  nparams = maxparam = 0;
  params = NULL;
  elem2param = NULL;
  map = NULL;

  maxshort = 10;
  neighshort = NULL;

  swpoly_c0=NULL;
  swpoly_c1=NULL;
  swpoly_c2=NULL;
  swpoly_c3=NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairSWPOLY::~PairSWPOLY()
{
  if (copymode) return;

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  memory->destroy(params);
  memory->destroy(elem2param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
    delete [] map;
  }
  memory->destroy(swpoly_c0);
  memory->destroy(swpoly_c1);
  memory->destroy(swpoly_c2);
  memory->destroy(swpoly_c3);
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum,jnumm1;
  int itype,jtype,ktype,ijparam,ikparam,ijkparam;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fj[3],fk[3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      if (rsq >= params[ijparam].cutsq) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }

      twobody(&params[ijparam],rsq,fpair,eflag,evdwl);

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;
      f[j][0] -= delx*fpair;
      f[j][1] -= dely*fpair;
      f[j][2] -= delz*fpair;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delx,dely,delz);
    }

    jnumm1 = numshort - 1;

    for (jj = 0; jj < jnumm1; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      double fjxtmp,fjytmp,fjztmp;
      fjxtmp = fjytmp = fjztmp = 0.0;

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        ikparam = elem2param[itype][ktype][ktype];
        ijkparam = elem2param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        threebody(&params[ijparam],&params[ikparam],&params[ijkparam],
                  rsq1,rsq2,delr1,delr2,fj,fk,eflag,evdwl);

        fxtmp -= fj[0] + fk[0];
        fytmp -= fj[1] + fk[1];
        fztmp -= fj[2] + fk[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        if (evflag) ev_tally3(i,j,k,evdwl,0.0,fj,fk,delr1,delr2);
      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSWPOLY::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSWPOLY::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1; // volta: 4 --> 2
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j; // volta: 4 --> 2
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  // read potential file and initialize potential parameters
  read_file(arg[2]);
  setup_params();
  // read poly coefficients
  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential("coeffs.swpoly");
    if (fp == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open polynomial coefficients file coeffs.swpoly. This file should be in your work directory!");
      error->one(FLERR,str);
    }
    fscanf(fp,"%d",&swpoly_pieces);
    fclose(fp);
  }
  memory->create(swpoly_c0,swpoly_pieces,"pair:swpoly_c0");
  memory->create(swpoly_c1,swpoly_pieces,"pair:swpoly_c1");
  memory->create(swpoly_c2,swpoly_pieces,"pair:swpoly_c2");
  memory->create(swpoly_c3,swpoly_pieces,"pair:swpoly_c3");
  read_file_coeffs();
 
  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSWPOLY::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Stillinger-Weber requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Stillinger-Weber requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSWPOLY::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::read_file(char *file)
{
  int params_per_line = 13;
  char **words = new char*[params_per_line+1];
  char *aux;

  memory->sfree(params);
  params = NULL;
  nparams = maxparam = 0;

  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open Stillinger-Weber potential file %s",file);
      error->one(FLERR,str);
    }
  }

  // read each set of params from potential file
  //   // one set of params can span multiple lines
  //     // store params if all 3 element tags are in element list

  int n,nwords,ielement,jelement,kelement;
  char line[MAXLINE],*ptr;
  int eof = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // concatenate additional lines until have params_per_line words

    while (nwords < params_per_line) {
      n = strlen(line);
      if (comm->me == 0) {
        ptr = fgets(&line[n],MAXLINE-n,fp);
        if (ptr == NULL) {
          eof = 1;
          fclose(fp);
        } else n = strlen(line) + 1;
      }
      MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof) break;
      MPI_Bcast(&n,1,MPI_INT,0,world);
      MPI_Bcast(line,n,MPI_CHAR,0,world);
      if ((ptr = strchr(line,'#'))) *ptr = '\0';
      nwords = atom->count_words(line);
    }

    if (nwords < params_per_line)
      error->all(FLERR,"Incorrect format in Stillinger-Weber potential file");

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    // ielement,jelement,kelement = 1st args
    //     // if all 3 args are in element list, then parse this line
    //         // else skip to next entry in file

    for (ielement = 0; ielement < nelements; ielement++)
      if (strcmp(words[0],elements[ielement]) == 0) break;
    if (ielement == nelements) continue;
    for (jelement = 0; jelement < nelements; jelement++)
      if (strcmp(words[1],elements[jelement]) == 0) break;
    if (jelement == nelements) continue;
    for (kelement = 0; kelement < nelements; kelement++)
      if (strcmp(words[2],elements[kelement]) == 0) break;
    if (kelement == nelements) continue;

    // load up parameter settings and error check their values

    if (nparams == maxparam) {
      maxparam += DELTA;
      params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                          "pair:params");
    }

    params[nparams].ielement = ielement;
    params[nparams].jelement = jelement;
    params[nparams].kelement = kelement;
    params[nparams].epsilon = atof(words[3]);
    params[nparams].sigma = atof(words[4]);
    params[nparams].littlea = atof(words[5]);
    params[nparams].lambda = atof(words[6]);
    params[nparams].gamma = atof(words[7]);
	// excluded costheta0, but need to be in file
    params[nparams].biga = atof(words[9]);
    params[nparams].bigb = atof(words[10]);
    params[nparams].powerp = atof(words[11]);
    params[nparams].powerq = atof(words[12]);
    params[nparams].tol = atof(words[13]);

    if (params[nparams].epsilon < 0.0 || params[nparams].sigma < 0.0 ||
        params[nparams].littlea < 0.0 || params[nparams].lambda < 0.0 ||
        params[nparams].gamma < 0.0 || params[nparams].biga < 0.0 ||
        params[nparams].bigb < 0.0 || params[nparams].powerp < 0.0 ||
        params[nparams].powerq < 0.0 || params[nparams].tol < 0.0)
      error->all(FLERR,"Illegal Stillinger-Weber parameter");

    nparams++;
  }
  delete [] words;
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::read_file_coeffs()
{
  // read coefficients for polynomial function
  char **words = new char*[4];
  int n,nwords,eof=0,linha=0;
  char line[MAXLINE],*ptr;
  int maxcoeffs = swpoly_pieces;
  int npieces = maxcoeffs = 0;
  FILE *fp;
  // open file on proc 0
  if (comm->me == 0) {
    fp = force->open_potential("coeffs.swpoly");
    if (fp == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open polynomial coefficients file coeffs.swpoly. This file should be in your work directory!");
      error->one(FLERR,str);
    }
  }
  // reading file
  while (1) {
	// get lines using proc 0
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      linha++;
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(&linha,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);
    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0 || linha == 1) continue;
    // if first line than read the number of pieces, otherwise concatenate
      // concatenate additional lines until have 4 words
      while (nwords < 4) {
        n = strlen(line);
        if (comm->me == 0) {
          ptr = fgets(&line[n],MAXLINE-n,fp);
          if (ptr == NULL) {
            eof = 1;
            fclose(fp);
          } else n = strlen(line) + 1;
        }
        MPI_Bcast(&eof,1,MPI_INT,0,world);
        if (eof) break;
        MPI_Bcast(&n,1,MPI_INT,0,world);
        MPI_Bcast(line,n,MPI_CHAR,0,world);
        if ((ptr = strchr(line,'#'))) *ptr = '\0';
        nwords = atom->count_words(line);
      }
    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;
    /*if (npieces == maxcoeffs) {
      maxcoeffs += DELTA;
      memory->create(swpoly_c0,swpoly_pieces,"pair:swpoly_c0");
      memory->create(swpoly_c1,swpoly_pieces,"pair:swpoly_c1");
      memory->create(swpoly_c2,swpoly_pieces,"pair:swpoly_c2");
      memory->create(swpoly_c3,swpoly_pieces,"pair:swpoly_c3");
	}*/
	swpoly_c0[npieces]=atof(words[0]);
    swpoly_c1[npieces]=atof(words[1]);
    swpoly_c2[npieces]=atof(words[2]);
    swpoly_c3[npieces]=atof(words[3]);
    npieces++;
  }
  delete [] words;
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::setup_params()
{
  int i,j,k,m,n;
  double rtmp;

  // set elem2param for all triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem2param);
  memory->create(elem2param,nelements,nelements,nelements,"pair:elem2param");
  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem2param[i][j][k] = n;
      }

  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter
  // (cut = a*sigma)

  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].sigma*params[m].littlea;

    rtmp = params[m].cut;
    if (params[m].tol > 0.0) {
      if (params[m].tol > 0.01) params[m].tol = 0.01;
      if (params[m].gamma < 1.0)
        rtmp = rtmp +
          params[m].gamma * params[m].sigma / log(params[m].tol);
      else rtmp = rtmp +
             params[m].sigma / log(params[m].tol);
    }
    params[m].cutsq = rtmp * rtmp;

    params[m].sigma_gamma = params[m].sigma*params[m].gamma;
    params[m].c1 = params[m].biga*params[m].epsilon *
      params[m].powerp*params[m].bigb *
      pow(params[m].sigma,params[m].powerp);
    params[m].c2 = params[m].biga*params[m].epsilon*params[m].powerq *
      pow(params[m].sigma,params[m].powerq);
    params[m].c3 = params[m].biga*params[m].epsilon*params[m].bigb *
      pow(params[m].sigma,params[m].powerp+1.0);
    params[m].c4 = params[m].biga*params[m].epsilon *
      pow(params[m].sigma,params[m].powerq+1.0);
    params[m].c5 = params[m].biga*params[m].epsilon*params[m].bigb *
      pow(params[m].sigma,params[m].powerp);
    params[m].c6 = params[m].biga*params[m].epsilon *
      pow(params[m].sigma,params[m].powerq);
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++) {
    rtmp = sqrt(params[m].cutsq);
    if (rtmp > cutmax) cutmax = rtmp;
  }
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::twobody(Param *param, double rsq, double &fforce,
                     int eflag, double &eng)
{
  double r,rinvsq,rp,rq,rainv,rainvsq,expsrainv;

  r = sqrt(rsq);
  rinvsq = 1.0/rsq;
  rp = pow(r,-param->powerp);
  rq = pow(r,-param->powerq);
  rainv = 1.0 / (r - param->cut);
  rainvsq = rainv*rainv*r;
  expsrainv = exp(param->sigma * rainv);
  fforce = (param->c1*rp - param->c2*rq +
            (param->c3*rp -param->c4*rq) * rainvsq) * expsrainv * rinvsq;
  if (eflag) eng = (param->c5*rp - param->c6*rq) * expsrainv;
}

/* ---------------------------------------------------------------------- */

void PairSWPOLY::threebody(Param *paramij, Param *paramik, Param *paramijk,
                       double rsq1, double rsq2,
                       double *delr1, double *delr2,
                       double *fj, double *fk, int eflag, double &eng)
{
  int indsp;
  double xsp,c0,c1,c2,c3;
  double r1,rainv1,gsrainv1;
  double r2,rainv2,gsrainv2;
  double cs,delcs,ddelcs,facrad,frad1,frad2;
  double facang,facang12,r12;

  r1 = sqrt(rsq1);
  rainv1 =r1 - paramij->cut;
  gsrainv1 = paramij->sigma_gamma / rainv1; // radial function to get the exponential of

  r2 = sqrt(rsq2);
  rainv2 = r2 - paramik->cut;
  gsrainv2 = paramik->sigma_gamma / rainv2;

  r12=r1*r2;
  cs = (delr1[0]*delr2[0] + delr1[1]*delr2[1] + delr1[2]*delr2[2])/r12; // cos(theta)
  indsp=floor(swpoly_pieces*(cs+1)/2);
  c0=swpoly_c0[indsp];
  c1=swpoly_c1[indsp];
  c2=swpoly_c2[indsp];
  c3=swpoly_c3[indsp];
  xsp=((cs+1)*swpoly_pieces-2*indsp)/swpoly_pieces;
  delcs=(c3*xsp*xsp + c2*xsp)*xsp + c1*xsp + c0;
  ddelcs=(3*c3*xsp + 2*c2)*xsp + c1;

  facrad = paramijk->lambda*delcs*exp(gsrainv1)*exp(gsrainv2); // three body term computed!

  // force
  facang = paramijk->lambda*ddelcs*exp(gsrainv1)*exp(gsrainv2);
  frad1 = facrad*gsrainv1/rainv1/r1+cs*facang/rsq1;
  frad2 = facrad*gsrainv2/rainv2/r2+cs*facang/rsq2;
  facang12 = facang/r12; //

  fj[0] = delr1[0]*frad1-delr2[0]*facang12;
  fj[1] = delr1[1]*frad1-delr2[1]*facang12;
  fj[2] = delr1[2]*frad1-delr2[2]*facang12;

  fk[0] = delr2[0]*frad2-delr1[0]*facang12;
  fk[1] = delr2[1]*frad2-delr1[1]*facang12;
  fk[2] = delr2[2]*frad2-delr1[2]*facang12;

  if (eflag) eng = facrad;
}
